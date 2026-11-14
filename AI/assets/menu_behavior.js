(function () {
    function closeOpenMenus(except) {
        document.querySelectorAll(".top-menu .menu-group[open]").forEach(function (menu) {
            if (menu !== except) {
                menu.open = false;
            }
        });
    }

    document.addEventListener("click", function (event) {
        const summary = event.target.closest(".top-menu .menu-group > summary");
        if (summary) {
            const menu = summary.parentElement;
            closeOpenMenus(menu.open ? null : menu);
            return;
        }
        if (event.target.closest(".menu-dropdown .Select")) {
            return;
        }
        closeOpenMenus(null);
    });

    ["keydown", "keyup"].forEach(function (eventName) {
        document.addEventListener(eventName, function (event) {
            if (
                event.key !== "Escape"
                && event.target.closest(".menu-dropdown .Select")
            ) {
                return;
            }
            window.setTimeout(function () {
                closeOpenMenus(null);
            }, 0);
        });
    });
})();
