# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for EmbodimentTag enum consistency with N1.7 checkpoint.

Ensures that:
- All pretrain/posttrain tags have matching entries in the N1.7
  EMBODIMENT_TAG_TO_PROJECTOR_INDEX.
- Removed N1.6 tags are no longer in the enum or configs.
- resolve() error messages categorize tags by usage (base model vs finetuned).
- reverse_lookup() maps tag values back to enum names.
- Tag category sets (PRETRAIN_TAGS, POSTTRAIN_TAGS, FINETUNE_ONLY_TAGS) are
  exhaustive and non-overlapping.
"""

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import (
    FINETUNE_ONLY_TAGS,
    POSTTRAIN_TAGS,
    PRETRAIN_TAGS,
    EmbodimentTag,
)
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import EMBODIMENT_TAG_TO_PROJECTOR_INDEX
import pytest


class TestEmbodimentTagResolve:
    """Verify that EmbodimentTag.resolve() works case-insensitively."""

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            # By enum name (various cases)
            ("xdof", EmbodimentTag.XDOF),
            ("XDOF", EmbodimentTag.XDOF),
            ("new_embodiment", EmbodimentTag.NEW_EMBODIMENT),
            ("NEW_EMBODIMENT", EmbodimentTag.NEW_EMBODIMENT),
            ("unitree_g1", EmbodimentTag.UNITREE_G1),
            ("real_g1", EmbodimentTag.REAL_G1),
            # By enum value (various cases)
            ("libero_sim", EmbodimentTag.LIBERO_PANDA),
            (
                "oxe_droid_relative_eef_relative_joint",
                EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
            ),
            (
                "xdof_relative_eef_relative_joint",
                EmbodimentTag.XDOF,
            ),
            (
                "real_g1_relative_eef_relative_joints",
                EmbodimentTag.REAL_G1,
            ),
            # Passthrough of existing enum
            (EmbodimentTag.XDOF, EmbodimentTag.XDOF),
            # Whitespace tolerance
            ("  xdof  ", EmbodimentTag.XDOF),
        ],
    )
    def test_resolve_known_tags(self, input_str, expected):
        assert EmbodimentTag.resolve(input_str) == expected

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown embodiment tag"):
            EmbodimentTag.resolve("nonexistent_robot")

    def test_resolve_error_lists_known_tags(self):
        with pytest.raises(ValueError, match="XDOF") as exc_info:
            EmbodimentTag.resolve("foo")
        msg = str(exc_info.value)
        for tag in EmbodimentTag:
            assert tag.name in msg

    def test_resolve_error_categorizes_tags(self):
        """Error message should separate base-model, posttrain, and finetuning tags."""
        with pytest.raises(ValueError) as exc_info:
            EmbodimentTag.resolve("fake_tag")
        msg = str(exc_info.value)
        assert "Base model tags" in msg
        assert "Posttrain tags" in msg
        assert "Finetuning-only tags" in msg


class TestReverseLookup:
    """Verify reverse_lookup maps tag values back to enum names."""

    def test_known_value(self):
        assert EmbodimentTag.reverse_lookup("xdof_relative_eef_relative_joint") == "XDOF"
        assert EmbodimentTag.reverse_lookup("libero_sim") == "LIBERO_PANDA"

    def test_unknown_value_returns_as_is(self):
        assert EmbodimentTag.reverse_lookup("some_internal_tag") == "some_internal_tag"


class TestTagCategories:
    """Verify PRETRAIN_TAGS, POSTTRAIN_TAGS, and FINETUNE_ONLY_TAGS are correct."""

    def test_categories_are_exhaustive(self):
        """Every enum member must be in exactly one category."""
        all_categorized = PRETRAIN_TAGS | POSTTRAIN_TAGS | FINETUNE_ONLY_TAGS
        for tag in EmbodimentTag:
            assert tag in all_categorized, (
                f"EmbodimentTag.{tag.name} is not in any category "
                f"(PRETRAIN_TAGS, POSTTRAIN_TAGS, or FINETUNE_ONLY_TAGS)"
            )

    def test_categories_are_non_overlapping(self):
        """No tag should appear in more than one category."""
        assert not (PRETRAIN_TAGS & POSTTRAIN_TAGS), (
            f"Overlap pretrain/posttrain: {PRETRAIN_TAGS & POSTTRAIN_TAGS}"
        )
        assert not (PRETRAIN_TAGS & FINETUNE_ONLY_TAGS), (
            f"Overlap pretrain/finetune: {PRETRAIN_TAGS & FINETUNE_ONLY_TAGS}"
        )
        assert not (POSTTRAIN_TAGS & FINETUNE_ONLY_TAGS), (
            f"Overlap posttrain/finetune: {POSTTRAIN_TAGS & FINETUNE_ONLY_TAGS}"
        )

    def test_new_embodiment_is_finetune_only(self):
        assert EmbodimentTag.NEW_EMBODIMENT in FINETUNE_ONLY_TAGS

    def test_g1_sonic_inspire_wrist_is_posttrain(self):
        assert EmbodimentTag.G1_SONIC_INSPIRE_WRIST in POSTTRAIN_TAGS

    def test_pretrain_tags_match_base_model(self):
        """Pretrain tags should match what's in the base model checkpoint."""
        expected_values = {
            "oxe_droid_relative_eef_relative_joint",
            "xdof_relative_eef_relative_joint",
            "xdof_relative_eef_relative_joint_subtask",
            "real_g1_relative_eef_relative_joints",
            "real_r1_pro_sharpa_relative_eef",
            "real_r1_pro_sharpa_relative_eef_human",
            "real_r1_pro_sharpa_relative_eef_maxinsights",
            "real_r1_pro_sharpa_relative_eef_mecka",
        }
        actual_values = {tag.value for tag in PRETRAIN_TAGS}
        assert actual_values == expected_values, (
            f"PRETRAIN_TAGS values don't match base model.\n"
            f"  Missing: {expected_values - actual_values}\n"
            f"  Extra: {actual_values - expected_values}"
        )


class TestRemovedN16Tags:
    """Verify that deprecated N1.6-only tags are fully removed."""

    @pytest.mark.parametrize("tag_name", ["OXE_GOOGLE", "OXE_WIDOWX", "GR1"])
    def test_removed_from_enum(self, tag_name):
        assert not hasattr(EmbodimentTag, tag_name), (
            f"EmbodimentTag.{tag_name} should be removed (not in N1.7 checkpoint)"
        )

    @pytest.mark.parametrize("tag_value", ["oxe_google", "oxe_widowx", "gr1_unified"])
    def test_removed_from_modality_configs(self, tag_value):
        assert tag_value not in MODALITY_CONFIGS, (
            f"MODALITY_CONFIGS['{tag_value}'] should be removed (not in N1.7 checkpoint)"
        )

    @pytest.mark.parametrize("tag_value", ["oxe_google", "oxe_widowx", "gr1_unified"])
    def test_removed_from_projector_index(self, tag_value):
        assert tag_value not in EMBODIMENT_TAG_TO_PROJECTOR_INDEX, (
            f"EMBODIMENT_TAG_TO_PROJECTOR_INDEX['{tag_value}'] should be removed"
        )


class TestEmbodimentTagConsistency:
    """Verify that all EmbodimentTag enum values have matching configs."""

    def test_all_tags_in_projector_index(self):
        """Every tag must have a projector index mapping."""
        for tag in EmbodimentTag:
            assert tag.value in EMBODIMENT_TAG_TO_PROJECTOR_INDEX, (
                f"EmbodimentTag.{tag.name} ('{tag.value}') missing from "
                f"EMBODIMENT_TAG_TO_PROJECTOR_INDEX"
            )

    def test_no_extra_projector_entries(self):
        """EMBODIMENT_TAG_TO_PROJECTOR_INDEX should not have orphan keys."""
        all_tag_values = {tag.value for tag in EmbodimentTag}
        for key in EMBODIMENT_TAG_TO_PROJECTOR_INDEX:
            assert key in all_tag_values, (
                f"EMBODIMENT_TAG_TO_PROJECTOR_INDEX has orphan key '{key}' "
                f"with no matching EmbodimentTag"
            )

    def test_no_extra_modality_config_entries(self):
        """MODALITY_CONFIGS should not have orphan keys without a matching EmbodimentTag."""
        all_tag_values = {tag.value for tag in EmbodimentTag}
        for key in MODALITY_CONFIGS:
            assert key in all_tag_values, (
                f"MODALITY_CONFIGS has orphan key '{key}' with no matching EmbodimentTag"
            )

    def test_posttrain_tags_with_builtin_configs_in_modality_configs(self):
        """Posttrain tags that need built-in modality configs should have them."""
        # These posttrain tags get configs from their finetuned checkpoint,
        # not from MODALITY_CONFIGS.
        checkpoint_config_tags = {
            EmbodimentTag.SIMPLER_ENV_GOOGLE,
            EmbodimentTag.SIMPLER_ENV_WIDOWX,
            EmbodimentTag.OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT,
        }
        for tag in POSTTRAIN_TAGS:
            if tag in checkpoint_config_tags:
                continue
            assert tag.value in MODALITY_CONFIGS, (
                f"EmbodimentTag.{tag.name} ('{tag.value}') is a posttrain tag "
                f"but missing from MODALITY_CONFIGS"
            )

    def test_g1_sonic_inspire_wrist_extends_inspire_with_wrist_cameras(self):
        base_config = MODALITY_CONFIGS[EmbodimentTag.UNITREE_G1_SONIC_INSPIRE.value]
        wrist_config = MODALITY_CONFIGS[EmbodimentTag.G1_SONIC_INSPIRE_WRIST.value]

        assert wrist_config["video"].modality_keys == ["ego_view", "left_wrist", "right_wrist"]
        assert wrist_config["video"].delta_indices == base_config["video"].delta_indices
        assert wrist_config["state"].modality_keys == base_config["state"].modality_keys
        assert wrist_config["action"].modality_keys == base_config["action"].modality_keys
        assert wrist_config["action"].delta_indices == base_config["action"].delta_indices
        assert wrist_config["language"].modality_keys == base_config["language"].modality_keys

    def test_unitree_g1_sonic_no_hand_uses_wrist_cameras_without_hand_modalities(self):
        no_hand_tag = EmbodimentTag.UNITREE_G1_SONIC_NO_HAND.value
        no_hand_config = MODALITY_CONFIGS[no_hand_tag]

        assert EMBODIMENT_TAG_TO_PROJECTOR_INDEX[no_hand_tag] == EMBODIMENT_TAG_TO_PROJECTOR_INDEX[
            EmbodimentTag.UNITREE_G1_SONIC.value
        ]
        assert no_hand_config["video"].modality_keys == ["ego_view", "left_wrist", "right_wrist"]
        assert no_hand_config["state"].modality_keys == [
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "projected_gravity",
        ]
        assert no_hand_config["action"].modality_keys == ["motion_token"]

    def test_unitree_g1_sonic_no_hand_wo_wrist_removes_only_wrist_cameras(self):
        base_tag = EmbodimentTag.UNITREE_G1_SONIC_NO_HAND.value
        wo_wrist_tag = EmbodimentTag.UNITREE_G1_SONIC_NO_HAND_WO_WRIST.value
        base_config = MODALITY_CONFIGS[base_tag]
        wo_wrist_config = MODALITY_CONFIGS[wo_wrist_tag]

        assert EMBODIMENT_TAG_TO_PROJECTOR_INDEX[wo_wrist_tag] == EMBODIMENT_TAG_TO_PROJECTOR_INDEX[
            base_tag
        ]
        assert wo_wrist_config["video"].modality_keys == ["ego_view"]
        assert wo_wrist_config["video"].delta_indices == base_config["video"].delta_indices
        assert wo_wrist_config["state"].modality_keys == base_config["state"].modality_keys
        assert wo_wrist_config["action"].modality_keys == base_config["action"].modality_keys
        assert wo_wrist_config["action"].delta_indices == base_config["action"].delta_indices
        assert wo_wrist_config["language"].modality_keys == base_config["language"].modality_keys
