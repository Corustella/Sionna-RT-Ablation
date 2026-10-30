"""Same-trace family maps, actual geometry, and bounded point-path diagnostics."""

import importlib

import numpy as np

from ablation_core import switches
from component_core import validate_components
from rt_backend import RTBackend


class ComponentBackend(RTBackend):
    def __init__(self, backend):
        super().__init__(backend)
        from component_solver import FamilyRadioMapSolver
        self.family_solver = FamilyRadioMapSolver()

    def solve(self, scene, measurement, samples, seed, max_depth, flags,
              total_samples=None, sample_offset=0):
        """Use the same FP64 reductions for the new dataset's switch ablations."""
        gain, _, _ = self.solve_components(scene, measurement, samples, seed, max_depth,
                                         total_samples, sample_offset, flags)
        return gain[0]

    def solve_components(self, scene, measurement, samples, seed, max_depth,
                         total_samples=None, sample_offset=0, flags=None):
        total_samples = samples if total_samples is None else total_samples
        module = importlib.import_module("sionna.rt.radio_map_solvers.radio_map_solver")
        original = module.fibonacci_lattice
        if total_samples != samples or sample_offset:
            module.fibonacci_lattice = lambda count: self.lattice_slice(total_samples, sample_offset, count)
        try:
            result = self.family_solver(scene, measurement_surface=measurement,
                samples_per_tx=samples, seed=seed, max_depth=max_depth, rr_depth=-1,
                stop_threshold=None, **(switches("full") if flags is None else flags))
            if result.num_tx != 1:
                raise ValueError("This dataset requires exactly one TX per terrain")
            gain = np.concatenate((result.path_gain.numpy().reshape(1, -1),
                                    result.family_gain.numpy().reshape(5, -1))).astype(np.float64)
            hits = np.asarray(result.hit_count.numpy(), np.uint64).reshape(5, -1).copy()
            positive_hits = np.asarray(result.positive_hit_count.numpy(), np.uint64).reshape(5, -1).copy()
            validate_components(gain, hits, positive_hits)
            return gain, hits, positive_hits
        finally:
            module.fibonacci_lattice = original

    def geometry(self, scene, measurement):
        def arrays(mesh):
            params = self.mi.traverse(mesh)
            return (np.asarray(params["vertex_positions"].numpy(), np.float32).reshape(-1, 3).copy(),
                    np.asarray(params["faces"].numpy(), np.uint32).reshape(-1, 3).copy())
        vertices, faces = arrays(scene.mi_scene.shapes()[0])
        measurement_vertices, measurement_faces = arrays(measurement)
        corners = measurement_vertices.astype(np.float64)[measurement_faces]
        centers = corners.mean(axis=1)
        areas = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0],
                                       corners[:, 2] - corners[:, 0]), axis=1) / 2
        if (areas <= 0).any():
            raise ValueError("Degenerate measurement triangles")
        tx = next(iter(scene.transmitters.values()))
        return {"terrain_vertices": vertices, "terrain_faces": faces,
                "measurement_vertices": measurement_vertices, "measurement_faces": measurement_faces,
                "cell_centers": centers, "cell_areas": areas,
                "tx_xyz": np.asarray(tx.position.numpy(), np.float32).reshape(3),
                "tx_orientation": np.asarray(tx.orientation.numpy(), np.float32).reshape(3)}

    def centroid_los(self, scene, centers):
        # Original terrain scene only: measurement surface must not occlude itself.
        tx = next(iter(scene.transmitters.values()))
        source = np.asarray(tx.position.numpy()).reshape(3)
        delta = np.asarray(centers, np.float32) - source
        distance = np.linalg.norm(delta, axis=1)
        if (distance <= 1e-4).any():
            raise ValueError("RX centroid coincides with TX")
        ray = self.mi.Ray3f(o=self.mi.Point3f(source.tolist()), d=self.mi.Vector3f((delta / distance[:, None]).T))
        ray.maxt = self.mi.Float(distance - 1e-4)
        return ~np.asarray(scene.mi_scene.ray_test(ray).numpy(), bool).reshape(-1)

    def point_paths(self, scene, positions, seed, samples, max_depth, max_paths):
        """Raw Paths tensors for explicitly modeled dual-polarized point receivers.

        The candidate cap is checked BEFORE shrink/image-method rejection. A
        capped run fails rather than being exported as complete. One TX only.
        This is a separate point-path experiment, not RadioMap ray records.
        """
        rt, dr = self.rt, self.dr
        if scene.receivers or len(scene.transmitters) != 1:
            raise ValueError("Point diagnostics require one TX and no existing receivers")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, vertical_spacing=.5,
                                       horizontal_spacing=.5, pattern="iso", polarization="VH")
        solver = rt.PathSolver()
        original = solver._candidate_generator
        candidate_count = []

        def checked_candidates(**kwargs):
            buffer = original(**kwargs)
            dr.eval(buffer.paths_counter)
            count = int(np.asarray(buffer.paths_counter.numpy()).max(initial=0))
            candidate_count.append(count)
            if count >= max_paths:
                raise RuntimeError(f"Path candidate cap reached ({count}/{max_paths}); "
                                   "increase --path-max-candidates or reduce --path-rx-batch in a new output")
            return buffer

        solver._candidate_generator = checked_candidates
        names = []
        try:
            for i, xyz in enumerate(positions):
                name = f"diagnostic_rx_{i}"
                scene.add(rt.Receiver(name=name, position=xyz.tolist(), orientation=[0, 0, 0]))
                names.append(name)
            paths = solver(scene, seed=seed, samples_per_src=samples,
                           max_num_paths_per_src=max_paths, max_depth=max_depth,
                           synthetic_array=True, **switches("full"))
            # With co-located 1x1 arrays, buffer coefficients equal the public
            # Paths.a coefficients (no synthetic element phase). Export compact
            # native records instead of padding every RX to the largest path
            # count and allocating depth x RX x max_paths tensors.
            buffer = paths._paths_buffer
            count = buffer.buffer_size
            if count:
                real = np.stack([np.stack([dr.real(a).numpy() for a in row]) for row in buffer.a]).transpose(2, 0, 1)
                imag = np.stack([np.stack([dr.imag(a).numpy() for a in row]) for row in buffer.a]).transpose(2, 0, 1)
            else:
                real, imag = np.zeros((0, 2, 1), np.float32), np.zeros((0, 2, 1), np.float32)
            result = {"a_real": real, "a_imag": imag,
                      "valid": buffer.valid.numpy().copy(), "interactions": buffer.interaction_types.numpy().copy(),
                      "vertices": np.stack((buffer.vertices_x.numpy(), buffer.vertices_y.numpy(), buffer.vertices_z.numpy()), axis=-1),
                      "delay_s": buffer.tau.numpy().copy(),
                      "objects": buffer.shapes.numpy().copy(), "primitives": buffer.primitives.numpy().copy(),
                      "rx_index": buffer.target_indices.numpy().copy(), "tx_index": buffer.source_indices.numpy().copy(),
                      "theta_t": buffer.theta_t.numpy().copy(), "phi_t": buffer.phi_t.numpy().copy(),
                      "theta_r": buffer.theta_r.numpy().copy(), "phi_r": buffer.phi_r.numpy().copy(),
                      "candidate_count_before_rejection": np.int64(candidate_count[0]),
                      "candidate_cap_reached": np.bool_(False)}
            # Object IDs are process-local shape pointers. Persist the mapping in
            # EVERY chunk, since resumed runs construct a new scene instance.
            result["object_mesh_names"] = np.asarray(list(scene.objects))
            result["object_runtime_ids"] = np.asarray([int(obj.object_id) for obj in scene.objects.values()], np.uint32)
            if not (np.isfinite(result["a_real"]).all() and np.isfinite(result["a_imag"]).all()):
                raise ValueError("Nonfinite path coefficient")
            if count:
                # Verify the compact coefficients against public Paths.a without
                # transferring its RX-padded tensor to the CPU.
                chosen = np.linspace(0, count-1, min(count, 64), dtype=np.uint32)
                rx = dr.gather(self.mi.UInt, buffer.target_indices, self.mi.UInt(chosen))
                path_index = dr.gather(self.mi.UInt, paths._path_ind, self.mi.UInt(chosen))
                public_real, public_imag = paths.a
                width = public_real.shape[-1]
                for pol in range(2):
                    index = (rx * 2 + pol) * width + path_index
                    for public, compact in ((public_real, real), (public_imag, imag)):
                        check = dr.gather(self.mi.Float, public.array, index).numpy()
                        if not np.allclose(check, compact[chosen, pol, 0], rtol=1e-6, atol=0):
                            raise ValueError("Compact/native coefficient differs from public Paths.a")
            return result
        finally:
            for name in names:
                scene.remove(name)
