import React, { useState, useEffect, useRef } from "react";

const App = () => {
    const [menuItems] = useState([
        { name: "Classic Margherita", price: "$9.99" },
        { name: "Pepperoni Feast", price: "$12.99" },
        { name: "Veggie Supreme", price: "$11.99" },
    ]);
    const [distance, setDistance] = useState("")
    const [squint, setSquint] = useState("")
    const [iris,setIris] = useState("")
    const [error, setError] = useState("");
    const [showCamera, setShowCamera] = useState(true);
    const [fontSize, setFontSize] = useState(16);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

    useEffect(() => {
        const videoElement = videoRef.current;
        console.log("here 1")
        navigator.mediaDevices
            .getUserMedia({ video: true })
            .then((stream) => {
                videoElement.srcObject = stream;
            })
            .catch((err) => {
                setError("Unable to access the camera.");
                console.error("Camera error:", err);
            });
        console.log("here 2")
        const intervalId = setInterval(() => {
            console.log("here 3")
            let response = captureAndSendImage();
            console.log(response)
        }, 6000);

        return () => clearInterval(intervalId);
    }, []);



    const updateScreen = (distanceNumber) => {
        if(distance !== ""){
            try{
                let newDistance = parseFloat(distanceNumber.replaceAll("cm"))
                setFontSize(newDistance/3)
            }catch(e){

            }
        }
    }

    const captureAndSendImage = async () => {
        if (!videoRef.current || !canvasRef.current) {
            console.error("Video or canvas reference is missing.");
            return;
        }

        const canvas = canvasRef.current;
        const video = videoRef.current;
        const context = canvas.getContext("2d");

        // Set canvas dimensions to match video feed
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw the current frame from the video onto the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to a blob (image format)
        canvas.toBlob(async (blob) => {
            if (!blob) {
                console.error("Failed to capture image blob.");
                return;
            }

            // Create FormData and append the image blob
            const formData = new FormData();
            formData.append("file", blob, "image.jpg");

            try {
                // Send the image to the API
                const response = await fetch("http://127.0.0.1:8080/Vision/evaluate", {
                    method: "POST",
                    body: formData,
                });

                if (response.ok) {
                    const data = await response.json();
                    updateScreen(data.distance)
                    setDistance(data.distance)
                    setSquint(data.squint)
                    setIris(data.iris)
                    console.log("Response from API:", data);
                    return data;
                } else {
                    console.error("API error:", response.status, response.statusText);
                }
            } catch (error) {
                console.error("Error sending image to API:", error);
            }
        }, "image/jpeg");
    };

    const themeStyles = {
        fontSize: `${fontSize}px`,
    };

    return (
        <div
            style={{...themeStyles, backgroundColor: "#121212", color: "#e0e0e0", minHeight: "100vh", padding: "1rem"}}>
            <header>
                <h1>Pizza Palace</h1>
                <nav>
                    <ul>
                        <li><a href="#menu">Menu</a></li>
                        <li><a href="#about">About Us</a></li>
                        <li><a href="#contact">Contact</a></li>
                        <li><a href="#order" className="button">Order Now</a></li>
                    </ul>
                </nav>
            </header>
            <section>
                <h2>The Best Pizza in Town!</h2>
                <p>Fresh ingredients, amazing flavors, made with love.</p>
                <a href="#menu" className="button">View Menu</a>
            </section>
            <section id="menu">
                <h2>Our Menu</h2>
                <div className="menu-grid">
                    {menuItems.map((item, index) => (
                        <div key={index} className="menu-item">
                            <h3>{item.name}</h3>
                            <p>{item.price}</p>
                        </div>
                    ))}
                </div>
            </section>
            <section id="about">
                <h2>About Us</h2>
                <p>At Pizza Palace, we pride ourselves on delivering the finest quality pizzas, crafted with fresh,
                    local ingredients and a passion for taste. Come taste the difference today!</p>
            </section>
            {error && <p style={{color: "red"}}>{error}</p>}
            <footer>
                <button onClick={() => setShowCamera(!showCamera)}>
                    {showCamera ? "Hide Camera" : "Show Camera"}
                </button>
                <button onClick={() => setFontSize(fontSize + 2)}>Increase Font Size</button>
                <button onClick={() => setFontSize(fontSize - 2)}>Decrease Font Size</button>
            </footer>
            {showCamera && (
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    style={{
                        position: "fixed",
                        bottom: "10px",
                        right: "10px",
                        width: "200px",
                        height: "200px",
                        border: "2px solid #e0e0e0",
                    }}
                />
            )}
            <canvas ref={canvasRef} style={{display: "none"}}/>
            <div className="box">
                <p>
                    <strong>Distance:</strong> {distance}
                </p>
                <p>
                    <strong>Squint:</strong> {squint}
                </p>
                <p>
                    <strong>Iris:</strong> {iris}
                </p>
            </div>
        </div>
    );
};

export default App;