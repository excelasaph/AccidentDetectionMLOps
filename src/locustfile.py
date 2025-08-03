from locust import HttpUser, task, between
import os
import random

class AccidentDetectionUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Set up test data paths"""
        self.test_images_dir = "data/test"
        self.accident_images = []
        self.non_accident_images = []
        
        # Collect available test images
        accident_dir = os.path.join(self.test_images_dir, "Accident")
        non_accident_dir = os.path.join(self.test_images_dir, "Non Accident")
        
        if os.path.exists(accident_dir):
            self.accident_images = [
                os.path.join(accident_dir, f) for f in os.listdir(accident_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ][:10]  # Limit to 10 images for testing
            
        if os.path.exists(non_accident_dir):
            self.non_accident_images = [
                os.path.join(non_accident_dir, f) for f in os.listdir(non_accident_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ][:10]  # Limit to 10 images for testing

    @task(3)
    def test_health_endpoint(self):
        """Test the health check endpoint - most frequent"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(2)
    def test_prediction_stats(self):
        """Test prediction statistics endpoint"""
        with self.client.get("/prediction-stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Prediction stats failed: {response.status_code}")

    @task(2)
    def test_training_data_sources(self):
        """Test training data sources endpoint"""
        with self.client.get("/training-data-sources", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Training data sources failed: {response.status_code}")

    @task(1)
    def test_prediction_history(self):
        """Test prediction history endpoint"""
        with self.client.get("/prediction-history?limit=10", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Prediction history failed: {response.status_code}")

    @task(1)
    def test_storage_stats(self):
        """Test storage statistics endpoint"""
        with self.client.get("/storage-stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Storage stats failed: {response.status_code}")

    @task(1)
    def test_predict_accident_image(self):
        """Test prediction with accident images"""
        if not self.accident_images:
            return
            
        # Select random accident image
        image_path = random.choice(self.accident_images)
        
        if os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as f:
                    files = {'files': (os.path.basename(image_path), f, 'image/jpeg')}
                    with self.client.post("/predict", files=files, catch_response=True) as response:
                        if response.status_code == 200:
                            result = response.json()
                            if 'prediction' in result:
                                response.success()
                            else:
                                response.failure("Missing prediction in response")
                        else:
                            response.failure(f"Prediction failed: {response.status_code}")
            except Exception as e:
                print(f"Error testing accident prediction: {e}")

    @task(1)
    def test_predict_non_accident_image(self):
        """Test prediction with non-accident images"""
        if not self.non_accident_images:
            return
            
        # Select random non-accident image
        image_path = random.choice(self.non_accident_images)
        
        if os.path.exists(image_path):
            try:
                with open(image_path, 'rb') as f:
                    files = {'files': (os.path.basename(image_path), f, 'image/jpeg')}
                    with self.client.post("/predict", files=files, catch_response=True) as response:
                        if response.status_code == 200:
                            result = response.json()
                            if 'prediction' in result:
                                response.success()
                            else:
                                response.failure("Missing prediction in response")
                        else:
                            response.failure(f"Prediction failed: {response.status_code}")
            except Exception as e:
                print(f"Error testing non-accident prediction: {e}")

# Custom user class for testing upload functionality
class DataUploadUser(HttpUser):
    wait_time = between(5, 10)  # Longer wait for upload tests
    weight = 1  # Lower frequency than main user

    @task
    def test_upload_data(self):
        """Test data upload functionality"""
        # This is a lighter test - just test the endpoint responsiveness
        # In real deployment, you'd want to test with actual files
        with self.client.post("/upload", 
                            data={'label': 'Accident'}, 
                            files={'files': ('test.jpg', b'fake_image_data', 'image/jpeg')},
                            catch_response=True) as response:
            if response.status_code in [200, 422]:  # 422 for invalid file is acceptable
                response.success()
            else:
                response.failure(f"Upload test failed: {response.status_code}")

# Performance test scenarios
class StressTestUser(HttpUser):
    """High-frequency user for stress testing"""
    wait_time = between(0.1, 0.5)  # Very short wait times
    weight = 2
    
    @task
    def rapid_health_checks(self):
        """Rapid health check requests"""
        self.client.get("/health")
    
    @task
    def rapid_stats_requests(self):
        """Rapid stats requests"""
        self.client.get("/prediction-stats")
