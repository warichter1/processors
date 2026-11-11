(function () {
    function scrollToLatestMessage(conversation) {
        window.requestAnimationFrame(function () {
            conversation.scrollTop = conversation.scrollHeight;
        });
    }

    function observeConversation() {
        const conversation = document.getElementById("conversation-view");
        if (!conversation) {
            return false;
        }

        const observer = new MutationObserver(function () {
            scrollToLatestMessage(conversation);
        });
        observer.observe(conversation, { childList: true, subtree: true });
        scrollToLatestMessage(conversation);
        return true;
    }

    if (!observeConversation()) {
        const mountObserver = new MutationObserver(function () {
            if (observeConversation()) {
                mountObserver.disconnect();
            }
        });
        mountObserver.observe(document.body, { childList: true, subtree: true });
    }
})();
