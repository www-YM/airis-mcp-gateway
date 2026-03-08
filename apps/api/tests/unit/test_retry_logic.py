"""Tests for retry logic in ProcessRunner."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import asyncio

from app.core.process_runner import ProcessRunner, ProcessConfig, ProcessState


class TestProcessRunnerRetry:
    """Test retry logic in ProcessRunner.call_tool()."""

    @pytest.fixture
    def config(self):
        """Create a test config."""
        return ProcessConfig(
            name="test-server",
            command="echo",
            args=["test"],
        )

    @pytest.fixture
    def runner(self, config):
        """Create a test runner."""
        return ProcessRunner(config)

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self, runner):
        """Successful call should not trigger retry."""
        runner.ensure_ready = AsyncMock(return_value=True)
        runner._send_request = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "success"}]}
        })

        result = await runner.call_tool("test_tool", {})

        assert "result" in result
        runner.ensure_ready.assert_called_once()
        runner._send_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_on_init_failure(self, runner):
        """Should retry when initialization fails."""
        # First call fails to init, second succeeds
        runner.ensure_ready = AsyncMock(side_effect=[False, True])
        runner._restart_process = AsyncMock()
        runner._send_request = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "success"}]}
        })

        result = await runner.call_tool("test_tool", {}, max_retries=2)

        assert "result" in result
        assert runner.ensure_ready.call_count == 2
        runner._restart_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, runner):
        """Should retry on request timeout."""
        runner.ensure_ready = AsyncMock(return_value=True)
        runner._restart_process = AsyncMock()
        # First call times out, second succeeds
        runner._send_request = AsyncMock(side_effect=[
            asyncio.TimeoutError(),
            {"jsonrpc": "2.0", "id": 1, "result": {"success": True}}
        ])

        with patch('app.core.process_runner.settings') as mock_settings:
            mock_settings.TOOL_CALL_TIMEOUT = 10.0
            result = await runner.call_tool("test_tool", {}, max_retries=2)

        assert "result" in result
        assert runner._send_request.call_count == 2
        runner._restart_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_retry_on_remote_internal_error(self, runner):
        """Should NOT retry on remote -32603 errors (subprocess responded normally).

        When _send_request returns (doesn't raise), it means the subprocess
        responded via STDIO. Even if the response contains a -32603 error,
        the process is healthy — the error was forwarded from a remote server
        (e.g. mcp-remote proxying a business error). No restart needed.
        """
        runner.ensure_ready = AsyncMock(return_value=True)
        runner._restart_process = AsyncMock()
        # Remote error forwarded through mcp-remote as -32603
        runner._send_request = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": "An error occurred"}
        })

        with patch('app.core.process_runner.settings') as mock_settings:
            mock_settings.TOOL_CALL_TIMEOUT = 10.0
            result = await runner.call_tool("test_tool", {}, max_retries=2)

        assert "error" in result
        assert result["error"]["code"] == -32603
        runner._send_request.assert_called_once()  # No retry
        runner._restart_process.assert_not_called()  # No restart

    @pytest.mark.asyncio
    async def test_no_retry_on_application_error(self, runner):
        """Should NOT retry on application-level errors (non -32603/-32000)."""
        runner.ensure_ready = AsyncMock(return_value=True)
        runner._restart_process = AsyncMock()
        # Return a "tool not found" style error
        runner._send_request = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "Method not found"}
        })

        with patch('app.core.process_runner.settings') as mock_settings:
            mock_settings.TOOL_CALL_TIMEOUT = 10.0
            result = await runner.call_tool("test_tool", {}, max_retries=2)

        assert "error" in result
        assert result["error"]["code"] == -32601
        runner._send_request.assert_called_once()  # No retry
        runner._restart_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, runner):
        """Should return error when max retries exceeded."""
        runner.ensure_ready = AsyncMock(return_value=False)
        runner._restart_process = AsyncMock()

        result = await runner.call_tool("test_tool", {}, max_retries=2)

        assert "error" in result
        assert "failed to initialize" in result["error"]["message"]
        assert runner.ensure_ready.call_count == 3  # Initial + 2 retries
        assert runner._restart_process.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_generic_exception(self, runner):
        """Should retry on generic exceptions."""
        runner.ensure_ready = AsyncMock(return_value=True)
        runner._restart_process = AsyncMock()
        # First call raises exception, second succeeds
        runner._send_request = AsyncMock(side_effect=[
            Exception("Connection lost"),
            {"jsonrpc": "2.0", "id": 1, "result": {"success": True}}
        ])

        with patch('app.core.process_runner.settings') as mock_settings:
            mock_settings.TOOL_CALL_TIMEOUT = 10.0
            result = await runner.call_tool("test_tool", {}, max_retries=2)

        assert "result" in result
        assert runner._send_request.call_count == 2
        runner._restart_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_retry_on_remote_server_error(self, runner):
        """Should NOT retry on -32000 errors from remote servers."""
        runner.ensure_ready = AsyncMock(return_value=True)
        runner._restart_process = AsyncMock()
        runner._send_request = AsyncMock(return_value={
            "jsonrpc": "2.0",
            "error": {"code": -32000, "message": "Server error from remote"}
        })

        with patch('app.core.process_runner.settings') as mock_settings:
            mock_settings.TOOL_CALL_TIMEOUT = 10.0
            result = await runner.call_tool("test_tool", {}, max_retries=2)

        assert "error" in result
        runner._send_request.assert_called_once()
        runner._restart_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_on_process_crash_exception(self, runner):
        """Should retry when _send_request raises (process crash/STDIO failure)."""
        runner.ensure_ready = AsyncMock(return_value=True)
        runner._restart_process = AsyncMock()
        # First call: process crashes (raises), second call: succeeds
        runner._send_request = AsyncMock(side_effect=[
            RuntimeError("Process not running"),
            {"jsonrpc": "2.0", "id": 1, "result": {"success": True}}
        ])

        with patch('app.core.process_runner.settings') as mock_settings:
            mock_settings.TOOL_CALL_TIMEOUT = 10.0
            result = await runner.call_tool("test_tool", {}, max_retries=2)

        assert "result" in result
        assert runner._send_request.call_count == 2
        runner._restart_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_zero_retries(self, runner):
        """Should not retry when max_retries=0."""
        runner.ensure_ready = AsyncMock(return_value=False)
        runner._restart_process = AsyncMock()

        result = await runner.call_tool("test_tool", {}, max_retries=0)

        assert "error" in result
        runner.ensure_ready.assert_called_once()
        runner._restart_process.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_process(self, runner):
        """Test _restart_process stops and allows restart."""
        runner.stop = AsyncMock()

        await runner._restart_process()

        runner.stop.assert_called_once()


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_starts_closed(self):
        """Circuit should start in CLOSED state."""
        from app.core.circuit import Circuit
        circuit = Circuit()
        assert circuit.state.state == "CLOSED"
        assert circuit.allow() == True

    def test_circuit_opens_on_failure(self):
        """Circuit should open after failure."""
        from app.core.circuit import Circuit
        circuit = Circuit(base_ms=1000)
        circuit.record_failure()
        assert circuit.state.state == "OPEN"

    def test_circuit_closes_on_success(self):
        """Circuit should close after success."""
        from app.core.circuit import Circuit
        circuit = Circuit()
        circuit.record_failure()
        assert circuit.state.state == "OPEN"
        circuit.record_success()
        assert circuit.state.state == "CLOSED"

    def test_circuit_exponential_backoff(self):
        """Circuit should use exponential backoff."""
        from app.core.circuit import Circuit
        import time
        circuit = Circuit(base_ms=1000, max_ms=30000)

        # First failure: ~1000ms backoff
        circuit.record_failure()
        retry1 = circuit.state.retry_at_ms

        # Reset and fail twice
        circuit.record_success()
        circuit.record_failure()
        circuit.record_failure()
        retry2 = circuit.state.retry_at_ms

        # Second failure should have longer backoff (approx 2x or more)
        # Note: jitter adds randomness, so we use a loose check
        assert retry2 > retry1

    def test_circuit_half_open(self):
        """Circuit should support half-open state."""
        from app.core.circuit import Circuit
        circuit = Circuit()
        circuit.half_open()
        assert circuit.state.state == "HALF_OPEN"
