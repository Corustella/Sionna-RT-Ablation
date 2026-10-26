"""Sionna RT 1.1.0 transport with indexed chunks of its global launch lattice."""

import importlib.metadata
import platform

import numpy as np


class RTBackend:
    def __init__(self, backend):
        version = importlib.metadata.version("sionna-rt")
        if version != "1.1.0":
            raise RuntimeError(f"Expected sionna-rt==1.1.0, found {version}")
        import mitsuba as mi
        mi.set_variant(f"{backend}_ad_mono_polarized")
        import drjit as dr
        import sionna.rt as rt
        self.mi, self.dr, self.rt = mi, dr, rt
        self.solver = rt.RadioMapSolver()
        self.versions = {name: importlib.metadata.version(name) for name in ("sionna-rt", "mitsuba", "drjit", "numpy", "scipy")}
        self.versions["mitsuba_variant"] = mi.variant()
        self.versions["python"] = platform.python_version()

    def mesh(self, name, vertices, faces, material):
        mi = self.mi
        props = mi.Properties()
        props["bsdf"] = mi.load_dict({"type": "holder-material"})
        mesh = mi.Mesh(name=name, vertex_count=len(vertices), face_count=len(faces),
                       props=props, has_vertex_normals=False, has_vertex_texcoords=False)
        mesh.bsdf().radio_material = material
        parameters = mi.traverse(mesh)
        parameters["vertex_positions"] = mi.Float(np.asarray(vertices, np.float32).ravel())
        parameters["faces"] = mi.UInt(np.asarray(faces, np.uint32).ravel())
        parameters.update()
        return mesh

    def scene(self, vertices, faces, height, tx_rc, metadata, band):
        rt, mi = self.rt, self.mi
        simulation = metadata["simulation"]
        source = simulation["material"]
        material = rt.RadioMaterial(
            name="lunar_regolith", thickness=source["thickness_m"],
            relative_permittivity=source["relative_permittivity"],
            conductivity=source["conductivity_s_per_m"][band],
            scattering_coefficient=source["scattering_coefficient"],
            xpd_coefficient=source["xpd_coefficient"],
            scattering_pattern=source["scattering_pattern"])
        ground = self.mesh("terrain", vertices, faces, material)
        scene = rt.Scene(mi.load_dict({"type": "scene", "terrain": ground}))
        scene.frequency = metadata["bands_hz"][band]
        scene.bandwidth = simulation["bandwidth_hz"][band]
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5,
                                       horizontal_spacing=.5, pattern="iso", polarization="V")
        row, col = map(int, tx_rc)
        spacing = metadata["resolution_m_per_px"]
        scene.add(rt.Transmitter(name="tx", position=[col * spacing, row * spacing,
                  float(height[row, col]) + simulation["tx_height_m_agl"]],
                  power_dbm=simulation["tx_power_dbm"]))
        shifted = vertices.copy()
        shifted[:, 2] += simulation["rx_height_m_agl"]
        measurement = self.mesh("measurement", shifted, faces, material)
        return scene, measurement

    def release_scene(self, scene, measurement):
        """Break C++ holder -> Python material -> scene ownership cycles.

        Call only after all results have synchronized to NumPy. The scene is
        finished and must not be used for another solve after this teardown.
        """
        measurement.bsdf().radio_material = None
        for shape in scene.mi_scene.shapes():
            shape.bsdf().radio_material = None

    def lattice_slice(self, total, offset, count):
        """Evaluate the upstream Fibonacci formula only at the requested indices.

        Same global point set as fibonacci_lattice(total), O(count) memory.
        Changing the seed alone would repeat a smaller launch lattice.
        Adapted from NVIDIA Sionna RT 1.1.0; see THIRD_PARTY_NOTICES.md.
        """
        if total < 2 or offset < 0 or count < 1 or offset + count > total:
            raise ValueError("Invalid global Fibonacci lattice slice")
        mi, dr = self.mi, self.dr
        start, denominator = mi.Float64(offset), mi.Float64(total - 1)
        dr.make_opaque(start, denominator)
        indices = dr.arange(mi.Float64, count) + start
        golden_ratio = (1. + dr.sqrt(mi.Float64(5.))) / 2.
        azimuth = indices / golden_ratio
        return mi.Point2f(azimuth - dr.floor(azimuth), indices / denominator)

    def solve(self, scene, measurement, samples, seed, max_depth, flags,
              total_samples=None, sample_offset=0):
        total_samples = samples if total_samples is None else total_samples
        # The v1.1.0 solver has no public launch-index API. Temporarily replace
        # only its lattice callable, restoring it even on failure. The runner
        # is intentionally single-threaded; no installed source is changed.
        module = importlib.import_module("sionna.rt.radio_map_solvers.radio_map_solver")
        original = module.fibonacci_lattice
        if total_samples != samples or sample_offset:
            module.fibonacci_lattice = lambda count: self.lattice_slice(total_samples, sample_offset, count)
        try:
            result = self.solver(scene, measurement_surface=measurement, samples_per_tx=samples,
                                 max_depth=max_depth, seed=seed, rr_depth=-1,
                                 stop_threshold=None, **flags)
            # Material transport is unchanged. The outer samples/total_samples
            # weight changes 4*pi/samples to the full-lattice quadrature weight.
            # This remains valid after diffuse events; RR/thresholds are off.
            return np.asarray(result.path_gain.numpy(), np.float64).reshape(-1)
        finally:
            module.fibonacci_lattice = original
