const uploadButton = document.getElementById("uploadButton");
const askButton = document.getElementById("askButton");
const historyButton = document.getElementById("historyButton");


uploadButton.addEventListener("click", async () => {
    const fileInput = document.getElementById("pdfFile");
    const uploadStatus = document.getElementById("uploadStatus");

    const file = fileInput.files[0];

    if (!file) {
        uploadStatus.textContent = "Please select a PDF file.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    uploadButton.disabled = true;
    uploadButton.textContent = "Uploading...";

    uploadStatus.textContent = "Uploading...";

    try {
        const response = await fetch(
            "http://127.0.0.1:8000/upload",
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        if (!response.ok) {
            uploadStatus.textContent =
                data.detail || "Upload failed.";

            return;
        }

        uploadStatus.textContent =
            `Uploaded successfully. Chunks created: ${data.chunks_created}`;

        const documentId =
            data.filename.replace(/\.pdf$/i, "");

        document.getElementById("documentId").value =
            documentId;

    } catch (error) {
        uploadStatus.textContent =
            "Could not connect to the API.";

    } finally {
        uploadButton.disabled = false;
        uploadButton.textContent = "Upload PDF";
    }
});


askButton.addEventListener("click", async () => {
    const documentId = document
        .getElementById("documentId")
        .value
        .trim();

    const conversationId = document
        .getElementById("conversationId")
        .value
        .trim();

    const question = document
        .getElementById("question")
        .value
        .trim();

    const answerElement =
        document.getElementById("answer");

    const sourcesElement =
        document.getElementById("sources");

    if (!documentId || !conversationId || !question) {
        answerElement.textContent =
            "Please fill in all fields.";

        sourcesElement.innerHTML = "";

        return;
    }

    askButton.disabled = true;
    askButton.textContent = "Thinking...";

    answerElement.textContent = "Thinking...";
    sourcesElement.innerHTML = "";

    try {
        const response = await fetch(
            "http://127.0.0.1:8000/ask",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question,
                    document_id: documentId,
                    conversation_id: conversationId
                })
            }
        );

        const data = await response.json();

        if (!response.ok) {
            answerElement.textContent =
                data.detail || "Something went wrong.";

            return;
        }

        answerElement.textContent =
            data.answer;

        if (
            data.sources &&
            data.sources.length > 0
        ) {
            sourcesElement.innerHTML =
                "<h3>Sources</h3>";

            data.sources.forEach((source) => {
                const item =
                    document.createElement("p");

                item.textContent =
                    `${source.source} - Page ${source.page}`;

                sourcesElement.appendChild(item);
            });
        }

    } catch (error) {
        answerElement.textContent =
            "Could not connect to the API.";

    } finally {
        askButton.disabled = false;
        askButton.textContent = "Ask Question";
    }
});


historyButton.addEventListener("click", async () => {
    const conversationId = document
        .getElementById("conversationId")
        .value
        .trim();

    const historyElement =
        document.getElementById("history");

    if (!conversationId) {
        historyElement.textContent =
            "Please enter a conversation ID.";

        return;
    }

    historyButton.disabled = true;
    historyButton.textContent = "Loading...";

    historyElement.textContent =
        "Loading history...";

    try {
        const response = await fetch(
            `http://127.0.0.1:8000/history/${conversationId}`
        );

        const data = await response.json();

        if (!response.ok) {
            historyElement.textContent =
                data.detail || "Could not load history.";

            return;
        }

        if (
            !data.messages ||
            data.messages.length === 0
        ) {
            historyElement.textContent =
                "No conversation history found.";

            return;
        }

        historyElement.innerHTML = "";

        data.messages.forEach(
            (message, index) => {
                const item =
                    document.createElement("div");

                item.innerHTML = `
                    <p>
                        <strong>
                            Question ${index + 1}:
                        </strong>
                        ${message.question}
                    </p>

                    <p>
                        <strong>
                            Answer:
                        </strong>
                        ${message.answer}
                    </p>

                    <hr>
                `;

                historyElement.appendChild(item);
            }
        );

    } catch (error) {
        historyElement.textContent =
            "Could not connect to the API.";

    } finally {
        historyButton.disabled = false;
        historyButton.textContent = "View History";
    }
});