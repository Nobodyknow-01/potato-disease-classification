import React, { useState } from 'react';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [dragging, setDragging] = useState(false);

  // Handle when the user drags a file over the dropzone
  const handleDragOver = (e) => {
    e.preventDefault(); // Prevent default behavior (allows drop)
    setDragging(true);
  };

  // Handle when the user leaves the dropzone
  const handleDragLeave = () => {
    setDragging(false);
  };

  // Handle when a file is dropped into the dropzone
  const handleDrop = (e) => {
    e.preventDefault(); // Prevent default behavior
    setDragging(false); // Reset dragging state
    const file = e.dataTransfer.files[0]; // Get the first file dropped
    if (file) {
      setImage(file); // Set the dropped file
    }
  };

  // Handle when the user manually uploads a file (click to select)
  const handleImageUpload = (e) => {
    setImage(e.target.files[0]);
  };

  // Handle submission and send the image to the backend
  const handleSubmit = async () => {
    if (!image) {
      alert("Please upload an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", image);

    try {
      const response = await fetch("http://localhost:8001/predict", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log(data); // handle classification result
    } catch (error) {
      console.error("Error uploading the image:", error);
    }
  };

  return (
    <div className="App">
      <div className="content">
        <header className="App-header">
          <h1>Potato Disease Classifier</h1>
        </header>

        <main className="App-main">
          <div className="container">
            <h2>Upload Potato Leaf Image</h2>
            <p>Classify your potato leaf into Early Blight, Late Blight, or Healthy conditions.</p>

            {/* Image Upload Area (Same for drag-and-drop and image display) */}
            <div
              className={`upload-area ${dragging ? 'active' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              {!image ? (
                <>
                  <p>{dragging ? "Release to upload" : "Drag and Drop an Image Here or Click to Select"}</p>
                  <input
                    type="file"
                    id="fileInput"
                    onChange={handleImageUpload}
                    style={{ display: 'none' }} // Hide the file input
                  />
                  <button
                    type="button"
                    onClick={() => document.getElementById("fileInput").click()}
                  >
                    Click to Select Image
                  </button>
                </>
              ) : (
                <img
                  src={URL.createObjectURL(image)}
                  alt="Preview"
                  className="image-preview"
                />
              )}
            </div>

            <button className="btn" onClick={handleSubmit}>Classify</button>
          </div>
        </main>

        <footer>
          &copy; 2024 Potato Disease Classifier. All rights reserved.
        </footer>
      </div>
    </div>
  );
}

export default App;
