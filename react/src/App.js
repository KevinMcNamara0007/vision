import React, { useState, useEffect, useRef } from "react";

const App = () => {
    const randomTechArticlesToday = [
        {
            title: "Google’s New AI Model Surpasses Expectations",
            description:
                "Tech enthusiasts are buzzing about Google's latest AI breakthrough, saying it could revolutionize search as we know it.",
            url: "https://news.google.com/articles/example-url-1",
            source: "Google News",
            publishedAt: "2025-01-23T09:45:00Z",
            imageUrl: "https://picsum.photos/400/600?random=1"
        },
        {
            title: "Chromebooks Get Major Battery Life Upgrade",
            description:
                "A new update promises to extend battery life on most Chromebooks by up to 20%.",
            url: "https://news.google.com/articles/example-url-2",
            source: "Google News",
            publishedAt: "2025-01-23T10:20:00Z",
            imageUrl: "https://picsum.photos/400/600?random=2"
        },
        {
            title: "Google Announces Quantum Computing Partnership",
            description:
                "Collaborating with top tech giants, Google aims to build a quantum ecosystem that will shape the future of cryptography.",
            url: "https://news.google.com/articles/example-url-3",
            source: "Google News",
            publishedAt: "2025-01-23T11:05:00Z",
            imageUrl: "https://picsum.photos/400/600?random=3"
        },
        {
            title: "Cloud Gaming Services See Rapid Growth",
            description:
                "Recent reports show cloud gaming usage skyrocketing, with Google Stadia updates being a key driver for new adopters.",
            url: "https://news.google.com/articles/example-url-4",
            source: "Google News",
            publishedAt: "2025-01-23T11:50:00Z",
            imageUrl: "https://picsum.photos/400/600?random=4"
        },
        {
            title: "Google to Launch Updated Smart Home Devices",
            description:
                "The next generation of Google Nest products promises better integration, voice recognition, and security features.",
            url: "https://news.google.com/articles/example-url-5",
            source: "Google News",
            publishedAt: "2025-01-23T12:15:00Z",
            imageUrl: "https://picsum.photos/400/600?random=5"
        }
    ];

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
        try{
            let newDistance = parseFloat(distanceNumber.replaceAll("cm"))
            console.log("Was able to parse float")
            setFontSize(newDistance/3)
        }catch(e){
            console.log("Unable to configure new font size")
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
                    updateScreen(data.distance.toFixed(4))
                    setDistance(data.distance.toFixed(4))
                    setSquint(data.squint.toFixed(4))
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
            <div className="random-tech-articles-container">
                <h1>Tech Articles Today</h1>
                <ul className="articles-list">
                    {randomTechArticlesToday.map((article, index) => (
                        <li key={index} className="article-item">
                            <h2>{article.title}</h2>
                            {article.imageUrl && (
                                <img
                                    src={article.imageUrl}
                                    alt={article.title}
                                    className="article-image"
                                />
                            )}
                            <p>{article.description}</p>
                            <a href={article.url} target="_blank" rel="noopener noreferrer">
                                Read more
                            </a>
                            <p>Source: {article.source}</p>
                            <p>Published At: {article.publishedAt}</p>
                        </li>
                    ))}
                </ul>
            </div>
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