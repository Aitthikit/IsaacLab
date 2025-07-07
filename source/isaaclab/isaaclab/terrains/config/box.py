# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg  
from ..sub_terrain_cfg import FlatPatchSamplingCfg
# FlatPatch_CFG = FlatPatchSamplingCfg(
#     num_patches= 3,
#     x_range = (-1.0,1.0)
#     y_range = (-1.0,1.0)
#     z_range = (0.6,0.6) 
# )

BOX_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshBoxTerrainCfg(
        box_height_range=(0.1,1.0),
        platform_width = 5,
        )
    },
)
"""Box terrains configuration."""

PIT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pits": terrain_gen.MeshPitTerrainCfg(
        pit_depth_range=(0.1,1.0),
        platform_width = 5.0,
        flat_patch_sampling = {
                "pits_target" : FlatPatchSamplingCfg(num_patches=10, patch_radius=0.5, max_height_diff=0.05)
        },
        )
    },
)


MULTI_TERRAINS_PLANE_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "plane": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.6,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                )
        },
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(-5.2,-0.1)
                                                )
        },
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(0.1,5.2)
                                                )
        },
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.25), platform_width=2.0, border_width=0.25,
            flat_patch_sampling = {
              "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(-5.2,-0.1)
                                                )
        },
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.25), platform_width=2.0, border_width=0.25,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(0.1,5.2)
                                                )
        },
        ),       
    },
)

MULTI_TERRAINS_BOX_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.8,
            box_height_range=(0.1,1.0),
            platform_width = 2,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                )
        },
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(-5.2,-0.1)
                                                )
        },
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(0.1,5.2)
                                                )
        },
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.1, 0.25), platform_width=2.0, border_width=0.25,
            flat_patch_sampling = {
              "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(-5.2,-0.1)
                                                )
        },
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.1, 0.25), platform_width=2.0, border_width=0.25,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(0.1,5.2)
                                                )
        },
        ),       
    },
)

MULTI_TERRAINS_PIT_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pits": terrain_gen.MeshPitTerrainCfg(
            proportion=0.8,
            pit_depth_range=(0.1,1.0),
            platform_width = 2.5,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                )
        },
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(-5.2,-0.1)
                                                )
        },
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(0.1,5.2)
                                                )
        },
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.25), platform_width=2.0, border_width=0.25,
            flat_patch_sampling = {
              "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(-5.2,-0.1)
                                                )
        },
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.1, 0.25), platform_width=2.0, border_width=0.25,
            flat_patch_sampling = {
                "target" : FlatPatchSamplingCfg(num_patches=10000,
                                                patch_radius=0.5,
                                                max_height_diff=1.00,
                                                x_range =(-8.00,8.00),
                                                y_range =(-8.00,8.00),
                                                z_range =(0.1,5.2)
                                                )
        },
        ),       
    },
)






