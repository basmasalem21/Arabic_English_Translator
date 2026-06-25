document.getElementById("translateBtn")
.addEventListener("click", async () => {

    const text = document.getElementById("inputText").value;

    const response = await fetch("/translate", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: text
        })
    });

    const data = await response.json();

    document.getElementById("outputText").value =
        data.translation;
});