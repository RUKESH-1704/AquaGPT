document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const imageUpload = document.getElementById('image-upload');
    const imageUploadContainer = document.getElementById('image-upload-container');
    
    let isProcessing = false;

    function addMessage(message, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
        messageDiv.textContent = message;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showImageUpload() {
        imageUploadContainer.style.display = 'block';
        userInput.disabled = true;
        sendButton.disabled = true;
    }

    function hideImageUpload() {
        imageUploadContainer.style.display = 'none';
        userInput.disabled = false;
        sendButton.disabled = false;
    }

    function disableInput() {
        userInput.disabled = true;
        sendButton.disabled = true;
        isProcessing = true;
    }

    function enableInput() {
        if (!imageUploadContainer.style.display || imageUploadContainer.style.display === 'none') {
            userInput.disabled = false;
            sendButton.disabled = false;
        }
        isProcessing = false;
    }

    async function handleImageUpload(file) {
        if (isProcessing) return;
        // Client-side validation: allow only png/jpg/jpeg
        const allowedTypes = ['image/png', 'image/jpeg'];
        const allowedExts = ['png', 'jpg', 'jpeg'];
        const ext = file.name.split('.').pop().toLowerCase();
        if (!allowedExts.includes(ext) || !allowedTypes.includes(file.type)) {
            addMessage('Please upload a valid image file (png, jpg, jpeg).', false);
            hideImageUpload();
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);

        disableInput();
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot-message';
        loadingDiv.textContent = 'Analyzing image...';
        chatMessages.appendChild(loadingDiv);

        try {
            const response = await fetch('/predict_disease', {
                method: 'POST',
                body: formData
            });

            let data;
            try {
                data = await response.json();
            } catch (e) {
                // non-JSON response (e.g., plain text error)
                const text = await response.text();
                chatMessages.removeChild(loadingDiv);
                addMessage(`Error: ${text}`, false);
                hideImageUpload();
                enableInput();
                return;
            }

            chatMessages.removeChild(loadingDiv);

            if (data.error) {
                addMessage(`Error: ${data.error}`, false);
            } else {
                addMessage(data.response, false);
            }
        } catch (error) {
            chatMessages.removeChild(loadingDiv);
            addMessage('Sorry, there was an error processing the image.', false);
        }

        hideImageUpload();
        enableInput();
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (!message || isProcessing) return;

        addMessage(message, true);
        userInput.value = '';
        disableInput();

        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message bot-message';
        loadingDiv.textContent = 'Thinking...';
        chatMessages.appendChild(loadingDiv);

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: message }),
            });

            const data = await response.json();
            chatMessages.removeChild(loadingDiv);

            if (data.error) {
                addMessage('Sorry, there was an error processing your request.', false);
            } else {
                addMessage(data.response, false);
                if (data.request_image) {
                    showImageUpload();
                }
            }
        } catch (error) {
            chatMessages.removeChild(loadingDiv);
            addMessage('Sorry, there was an error connecting to the server.', false);
        }

        enableInput();
    }

    // Handle image upload
    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleImageUpload(file);
        }
    });

    // Send message on button click
    sendButton.addEventListener('click', sendMessage);

    // Send message on Enter key
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !isProcessing) {
            sendMessage();
        }
    });
});
