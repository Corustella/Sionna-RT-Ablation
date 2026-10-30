# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sionna RT 1.1.0 loop with mutually exclusive path-history accumulation.

Adapted from the frozen upstream tag (38ff0af9). Transport, RNG draws and
normalization are unchanged. Changes: history bits, classified accumulation,
and dynamic lookup of the launch lattice. See THIRD_PARTY_NOTICES.md.
Import only after selecting a Mitsuba variant. Single-threaded use only.
"""

import importlib
from typing import Callable, List, Tuple

import drjit as dr
import mitsuba as mi
from sionna.rt import Scene
from sionna.rt.antenna_pattern import antenna_pattern_to_world_implicit
from sionna.rt.constants import InteractionType
from sionna.rt.radio_map_solvers.mesh_radio_map import MeshRadioMap
from sionna.rt.radio_map_solvers.planar_radio_map import PlanarRadioMap
from sionna.rt.radio_map_solvers.radio_map import RadioMap
from sionna.rt.radio_map_solvers.radio_map_solver import RadioMapSolver
from sionna.rt.utils import rotation_matrix, spawn_ray_from_sources

upstream = importlib.import_module("sionna.rt.radio_map_solvers.radio_map_solver")


class FamilyMeshRadioMap(MeshRadioMap):
    def __init__(self, scene, meas_surface):
        super().__init__(scene, meas_surface)
        # Dense angular chunks can deposit tens of thousands of contributions
        # into one cell. Separate FP32 atomic sums then drift beyond closure
        # tolerance depending on GPU scheduling. Keep transport/weights in FP32,
        # but accumulate BOTH the independent total and families in FP64.
        tensor64 = dr.tensor_t(mi.Float64)
        self._pathgain_map = dr.zeros(tensor64, [self.num_tx, self.cells_count])
        self.family_gain = dr.zeros(tensor64, [5, self.num_tx, self.cells_count])
        count = 5 * self.num_tx * self.cells_count
        self.hit_count = dr.zeros(mi.UInt64, count)
        self.positive_hit_count = dr.zeros(mi.UInt64, count)

    def add(self, e_fields, solid_angle, array_w, si_mp, k_world, tx_indices, hit, history):
        # Same contribution and cell normalization as upstream MeshRadioMap.add.
        tensor_ind = tx_indices * self.cells_count + si_mp.prim_index
        a = dr.zeros(mi.Vector4f, 1)
        for e_field, aw in zip(e_fields, array_w):
            a += aw @ e_field
        a = dr.squared_norm(a)
        cos_theta = dr.abs(dr.dot(si_mp.n, k_world))
        w = solid_angle * dr.rcp(cos_theta)
        surface = self.measurement_surface
        prim_index = surface.face_indices(si_mp.prim_index, active=hit)
        v0 = surface.vertex_position(prim_index[0])
        v1 = surface.vertex_position(prim_index[1]) - v0
        v2 = surface.vertex_position(prim_index[2]) - v0
        cell_area = .5 * dr.sqrt(dr.squared_norm(v1) * dr.squared_norm(v2)
                                - dr.square(dr.dot(v1, v2)))
        w *= dr.rcp(cell_area)
        a *= w
        contribution = mi.Float64(a)
        dr.scatter_reduce(dr.ReduceOp.Add, self._pathgain_map.array, contribution, tensor_ind, active=hit)

        has_diffuse = (history & int(InteractionType.DIFFUSE)) != 0
        has_trans = (history & int(InteractionType.REFRACTION)) != 0
        # 0:LOS, 1:pure specular, 2:diffuse/no transmission,
        # 3:transmission/no diffuse, 4:diffuse AND transmission.
        family = dr.select(has_diffuse, dr.select(has_trans, 4, 2),
                           dr.select(has_trans, 3, dr.select(history == 0, 0, 1)))
        index = family * (self.num_tx * self.cells_count) + tensor_ind
        dr.scatter_reduce(dr.ReduceOp.Add, self.family_gain.array, contribution, index, active=hit)
        dr.scatter_reduce(dr.ReduceOp.Add, self.hit_count, mi.UInt64(1), index, active=hit)
        dr.scatter_reduce(dr.ReduceOp.Add, self.positive_hit_count, mi.UInt64(1), index,
                          active=hit & (a > 0))

    def finalize(self):
        # Compute the same FP32 scaling as upstream, then retain FP64 map sums.
        scaling = mi.Float64(dr.square(self._wavelength * dr.rcp(4. * dr.pi)))
        self._pathgain_map *= scaling
        self.family_gain *= scaling


class FamilyRadioMapSolver(RadioMapSolver):
    def __call__(self, *args, **kwargs):
        if kwargs.get("measurement_surface") is None:
            raise ValueError("Family solver requires a mesh measurement surface")
        original = upstream.MeshRadioMap
        upstream.MeshRadioMap = FamilyMeshRadioMap
        try:
            return super().__call__(*args, **kwargs)
        finally:
            upstream.MeshRadioMap = original

    @dr.syntax
    def _shoot_and_bounce(
        self,
        scene : Scene,
        radio_map : RadioMap,
        sampler : mi.Sampler,
        tx_positions : mi.Point3f,
        tx_orientations : mi.Point3f,
        tx_antenna_patterns : List[Callable[[mi.Float, mi.Float],
                                            Tuple[mi.Complex2f, mi.Complex2f]]],
        precoding_vec : Tuple[mi.TensorXf, mi.TensorXf],
        rel_ant_positions_tx : mi.Point3f,
        samples_per_tx : int,
        max_depth : int,
        los : bool,
        specular_reflection : bool,
        diffuse_reflection : bool,
        refraction : bool,
        rr_depth : int,
        rr_prob : float,
        stop_threshold : float | None,
    ) -> None:
        r"""
        Executes the shoot-and-bounce loop

        The ``radio_map`` is updated in-place.

        :param scene: Scene for which to compute the radio map
        :param radio_map: Radio map
        :param sampler: Sampler used to generate random numbers and vectors
        :param tx_positions: Positions of the transmitters
        :param tx_orientations: Orientations of the transmitters
        :param tx_antenna_patterns: Antenna pattern of the transmitters
        :param precoding_vec: Real and imaginary components of the
            complex-valued precoding vector
        :param rel_ant_positions_tx: Positions of the antenna elements relative
            to the transmitters positions
        :param samples_per_tx: Number of samples per source
        :param max_depth: Maximum depth
        :param los: If set to `True`, then the line-of-sight paths are computed
        :param specular_reflection: If set to `True`, then the specularly
            reflected paths are computed
        :param diffuse_reflection: If set to `True`, then the diffusely
            reflected paths are computed
        :param refraction: If set to `True`, then the refracted paths are
            computed
        :param rr_depth: Depth from which to start Russian roulette
        :param rr_prob: Minimum probability with which to keep a path when
            Russian roulette is enabled
        :param stop_threshold: Gain threshold (linear scale) below which a path
            is deactivated
        """

        num_txs = dr.shape(tx_positions)[1]
        num_samples = samples_per_tx * num_txs
        num_tx_ant_patterns = len(tx_antenna_patterns)

        tx_indices = dr.repeat(dr.arange(mi.UInt, num_txs), samples_per_tx)

        # Spawn rays from the transmitters
        ray = spawn_ray_from_sources(upstream.fibonacci_lattice, samples_per_tx,
                                     tx_positions)

        # Weights to account for the antenna array and precoding
        array_w = self._synthetic_array_weighting(scene, ray.d,
                                                  rel_ant_positions_tx,
                                                  precoding_vec)

        # Mask indicating which rays are active
        active = dr.full(dr.mask_t(mi.Float), True, num_samples)

        # Length of the ray tube.
        # Only required if a threshold is set to deactivate path based on their
        # gain
        if stop_threshold is not None:
            ray_tube_length = dr.zeros(mi.Float, num_samples)
        else:
            ray_tube_length = None

        # Solid angle of the ray tube.
        # It is required to compute the diffusely reflected field.
        # Initialized assuming that all the rays initially spawn from the source
        # share the unit sphere equally, i.e., initialized to
        # 4*PI/samples_per_src
        solid_angle = dr.full(mi.Float, 4.0 * dr.pi * dr.rcp(samples_per_tx),
                              num_samples)
        # Initialize the electric field
        # Orientation of sources and targets and correspond to-world transforms
        sample_tx_orientation = dr.repeat(tx_orientations, samples_per_tx)
        tx_to_world = rotation_matrix(sample_tx_orientation)
        # The following loop transports and update the electric field.
        # The electric field is initialized by the source antenna pattern
        e_fields = [antenna_pattern_to_world_implicit(src_antenna_pattern,
                                                      tx_to_world, ray.d,
                                                      direction="out")
                    for src_antenna_pattern in tx_antenna_patterns]

        history = dr.zeros(mi.UInt, num_samples)
        depth = mi.UInt(0)
        while dr.hint(active, mode=self.loop_mode, exclude=[array_w]):

            # Test intersection with the scene
            si_scene = scene.mi_scene.ray_intersect(ray, active=active)

            # Each interaction processes either an interaction with the scene
            # or an interaction with the measurement plane.
            # `mp_int` is set to `True` if the interaction is with the measurement
            # plane.
            # `scene_int` is set to `True` if the interaction is with the scene.
            # If the radio map is planar, then test interaction with the measurement
            # plane. Otherwise, the measurement surface is assumed to be part of
            # the scene and we only need to check that it has hit the measurement surface.
            si_mp = None
            if isinstance(radio_map, PlanarRadioMap):
                si_mp = radio_map.measurement_surface.ray_intersect(ray,
                                                                active=active)
                # An intersection with the measurement plane is valid only if
                # (i) it was not obstructed by the scene, and (ii) the intersection
                # is valid.
                mp_int = active & (si_mp.t < si_scene.t) & si_mp.is_valid()
                # Ignore interaction with measurement plane if LoS is disabled
                mp_int &= ((depth > 0) | los)
            else:
                # The measurement surface is part of the scene
                si_mp = si_scene
                mp_int = si_mp.shape == radio_map.measurement_surface

            # Update the radio map
            # The radio map is not updated if LoS is disabled and it is a
            # LoS interaction with the measurement surface
            # Note that we need to redo the LoS test to handle the case where
            # the measurement surface is part of the scene
            update_radio_map = mp_int & ((depth > 0) | los)
            radio_map.add(e_fields, solid_angle, array_w, si_mp, ray.d,
                          tx_indices, update_radio_map, history)

            # Update the active state of rays
            # Active if has hit the scene (which includes the measurement surface
            # if it is part of the scene)
            active &= si_scene.is_valid()

            # It is an interaction with the scene if (i) the ray has hit the scene
            # and (ii) it has not hit the measurement plane
            scene_int = active & ~mp_int

            # Sample the BSDF
            # Sample random numbers to sample the BSDF
            sample1 = sampler.next_1d()
            sample2 = sampler.next_2d()
            s, e_fields = self._sample_radio_material(
                si_scene, ray.d, e_fields,
                solid_angle, sample1, sample2, specular_reflection,
                diffuse_reflection, refraction, scene_int
            )
            interaction_type = dr.select(scene_int, s.sampled_component,
                                         InteractionType.NONE)
            history = dr.select(scene_int, history | interaction_type, history)
            diffuse = scene_int & (interaction_type == InteractionType.DIFFUSE)
            # Update the solid angle
            # If a diffuse reflection is sampled, then it is set to 2PI.
            # Otherwise it is left unchanged
            solid_angle = dr.select(diffuse, dr.two_pi, solid_angle)

            # Spawn rays for next iteration
            ray_scene_int = si_scene.spawn_ray(d=s.wo)
            ray_mp_int = si_mp.spawn_ray(d=ray.d)
            ray = dr.select(scene_int, ray_scene_int, ray_mp_int)
            depth = dr.select(scene_int, depth + 1, depth)
            active &= (depth <= max_depth)
            active &= mp_int | (interaction_type != InteractionType.NONE)

            ## Russian roulette and gain threshold deactivation

            # Compute the path gain (without applying the spreading factor)
            gain = dr.sum([dr.squared_norm(e_field) for e_field in e_fields])
            gain /= num_tx_ant_patterns

            # Update the ray tube length
            if stop_threshold is not None:
                ray_tube_length = dr.select(diffuse, 0.0, ray_tube_length)
                ray_tube_length += dr.select(scene_int, si_scene.t, si_mp.t)

            # Is the depth threshold to activate Russian roulette reached?
            rr_inactive = depth < rr_depth
            # User specify a maximum probability of continuing tracing
            rr_continue_prob = dr.minimum(gain, rr_prob)
            # Randomly stop tracing of rays
            rr_continue = scene_int & (sampler.next_1d() < rr_continue_prob)
            active &= (rr_inactive | rr_continue)

            # Scale the remaining rays accordingly to ensure an unbiased result
            for i in range(num_tx_ant_patterns):
                e_fields[i] = dr.select(rr_inactive, e_fields[i],
                                        e_fields[i] * dr.rsqrt(rr_continue_prob))

            # Deactivate rays with gains below the set threshold
            if stop_threshold is not None:
                gain_pl = gain*dr.square(scene.wavelength
                                         * dr.rcp(4. * dr.pi * ray_tube_length))
                th_continue = scene_int & (gain_pl > stop_threshold)
                active &= th_continue

        # Finalizes the computation of the radio maps
        radio_map.finalize()
