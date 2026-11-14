(function () {
    const pollIntervalMs = 5000;
    const requestTimeoutMs = 3000;
    let instanceId = null;
    let sawUnavailableServer = false;
    let requestInProgress = false;

    async function checkServer() {
        if (requestInProgress) {
            return;
        }
        requestInProgress = true;

        const controller = new AbortController();
        const timeout = window.setTimeout(
            function () {
                controller.abort();
            },
            requestTimeoutMs
        );

        try {
            const response = await fetch("/healthz", {
                cache: "no-store",
                credentials: "same-origin",
                signal: controller.signal
            });
            if (!response.ok) {
                throw new Error("Health check failed");
            }

            const health = await response.json();
            if (
                health.status !== "ok"
                || typeof health.instance_id !== "string"
                || !health.instance_id
            ) {
                throw new Error("Invalid health response");
            }

            if (
                sawUnavailableServer
                || (instanceId !== null && instanceId !== health.instance_id)
            ) {
                window.location.reload();
                return;
            }
            instanceId = health.instance_id;
        } catch (_error) {
            sawUnavailableServer = true;
        } finally {
            window.clearTimeout(timeout);
            requestInProgress = false;
        }
    }

    checkServer();
    window.setInterval(checkServer, pollIntervalMs);
})();
