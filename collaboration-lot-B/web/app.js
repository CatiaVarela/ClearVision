const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const previewImage = document.getElementById("preview");
const detectionsList = document.getElementById("detections");
const statusElement = document.getElementById("status");
const fastModeCheckbox = document.getElementById("fast-mode");

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));

dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragover");
  const droppedFile = event.dataTransfer.files[0];
  if (droppedFile) analyzeImage(droppedFile);
});

fileInput.addEventListener("change", () => {
  const selectedFile = fileInput.files[0];
  if (selectedFile) analyzeImage(selectedFile);
});

async function analyzeImage(imageFile) {
  statusElement.textContent = "Analyse en cours...";
  detectionsList.innerHTML = "";
  previewImage.style.display = "none";

  const formData = new FormData();
  formData.append("file", imageFile);

  const fastParameter = fastModeCheckbox.checked ? "?fast=true" : "";

  try {
    const response = await fetch(`/detect${fastParameter}`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Erreur serveur : ${response.status}`);
    }

    const result = await response.json();

    previewImage.src = `data:image/jpeg;base64,${result.annotated_image_base64}`;
    previewImage.style.display = "block";

    statusElement.textContent = `${result.detection_count} détection(s) trouvée(s)`;

    result.detections.forEach((detection) => {
      const listItem = document.createElement("li");
      const scoreText = detection.score ? ` (${(detection.score * 100).toFixed(0)} %)` : "";
      listItem.textContent = `${detection.label_fr}${scoreText}`;
      detectionsList.appendChild(listItem);
    });
  } catch (error) {
    statusElement.textContent = `Erreur : ${error.message}`;
  }
}
