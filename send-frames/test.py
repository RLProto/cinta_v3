import aiohttp
import asyncio
import os

async def upload_model(session, url, file_path):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = await session.post(url, data=files)
        if response.status == 200:
            print("Model uploaded successfully!")
            return await response.text()
        else:
            print(f"Failed to upload model: {response.status}")
            return None

async def send_image(session, url, image_path):
    with open(image_path, 'rb') as file:
        files = {'file': file}
        response = await session.post(url, data=files)
        if response.status == 200:
            print("Image sent successfully!")
            return await response.text()
        else:
            print(f"Failed to send image: {response.status}")
            return None

async def main():
    api_url = "http://localhost:9999"
    model_path = "D:/Python/cinta_v3/send-image/model/ag_v01_cloud.zip"
    image_path = "D:/Python/cinta_v3/send-image/model/image1.jpg"
    
    async with aiohttp.ClientSession() as session:
        # Upload the model
        model_response = await upload_model(session, f"{api_url}/upload-model", model_path)
        print("Model Upload Response:", model_response)

        # Send the image for inference
        image_response = await send_image(session, f"{api_url}/inference", image_path)
        print("Image Inference Response:", image_response)

if __name__ == "__main__":
    asyncio.run(main())
