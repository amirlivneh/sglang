"""Unit tests for TokenizerControlMixin request normalization."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sglang.srt.managers import tokenizer_control_mixin
from sglang.srt.managers.io_struct import (
    FlushCacheReqOutput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
)
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _QuietProfilePlatform(SRTPlatform):
    def apply_profile_request_defaults(self, req: ProfileReq) -> None:
        if req.with_stack is None:
            req.with_stack = False
        if req.record_shapes is None:
            req.record_shapes = False
        if req.num_steps is None:
            req.num_steps = 2


class TestStartProfilePlatformDefaults(CustomTestCase):
    def _start_profile(self, req):
        manager = MagicMock()
        manager._execute_profile = AsyncMock(
            return_value=ProfileReqOutput(success=True, message="")
        )
        with (
            patch.object(
                tokenizer_control_mixin, "current_platform", _QuietProfilePlatform()
            ),
            patch.dict(
                "os.environ",
                {
                    "SGLANG_PROFILE_WITH_STACK": "true",
                    "SGLANG_PROFILE_RECORD_SHAPES": "true",
                },
            ),
        ):
            asyncio.run(TokenizerControlMixin.start_profile(manager, req))
        return manager._execute_profile.call_args.args[0]

    def test_platform_sees_unset_fields_before_env_defaults(self):
        """Platform defaults apply only to fields the caller left unset;
        the env defaults must not fill them first."""
        for req in (None, ProfileReq()):
            with self.subTest(req=req):
                sent = self._start_profile(req)
                self.assertEqual(sent.req_type, ProfileReqType.START_PROFILE)
                self.assertFalse(sent.with_stack)
                self.assertFalse(sent.record_shapes)
                self.assertEqual(sent.num_steps, 2)


class TestFlushCacheResults(CustomTestCase):
    def test_partial_failure_is_reported_for_either_reply_order(self):
        succeeded = FlushCacheReqOutput(success=True)
        failed = FlushCacheReqOutput(success=False, message="rank busy")
        for results in ([succeeded, failed], [failed, succeeded]):
            with self.subTest(results=results):
                manager = MagicMock()
                manager.flush_cache_communicator = AsyncMock(return_value=results)

                response = asyncio.run(
                    TokenizerControlMixin.flush_cache(manager, timeout_s=5)
                )

                self.assertFalse(response.success)
                self.assertEqual(response.message, "rank busy")
                manager.mm_processor.clear_preprocess_cache.assert_not_called()
                request = manager.flush_cache_communicator.await_args.args[0]
                self.assertEqual(request.timeout_s, 5)

    def test_all_successes_clear_preprocess_cache(self):
        manager = MagicMock()
        manager.flush_cache_communicator = AsyncMock(
            return_value=[FlushCacheReqOutput(success=True) for _ in range(2)]
        )

        response = asyncio.run(TokenizerControlMixin.flush_cache(manager))

        self.assertTrue(response.success)
        manager.mm_processor.clear_preprocess_cache.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
