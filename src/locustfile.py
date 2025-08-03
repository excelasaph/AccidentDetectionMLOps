from locust import HttpUser, task, between
import os

class SimpleLoadTestUser(HttpUser):
    wait_time = between(2, 5)  # Increased wait time for remote server
    
    @task(5)
    def test_health_endpoint(self):
        """Test the health check endpoint - most frequent"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
                print(f"Health check successful: {response.status_code}")
            else:
                response.failure(f"Health check failed: {response.status_code}")
                print(f"Health check failed: {response.status_code}")

    @task(3)
    def test_prediction_stats(self):
        """Test prediction statistics endpoint"""
        with self.client.get("/prediction-stats", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
                print(f"Prediction stats successful: {response.status_code}")
            else:
                response.failure(f"Prediction stats failed: {response.status_code}")
                print(f"Prediction stats failed: {response.status_code}")

    @task(2)
    def test_training_data_sources(self):
        """Test training data sources endpoint"""
        with self.client.get("/training-data-sources", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
                print(f"Training data sources successful: {response.status_code}")
            else:
                response.failure(f"Training data sources failed: {response.status_code}")
                print(f"Training data sources failed: {response.status_code}")

    @task(1)
    def test_prediction_history(self):
        """Test prediction history endpoint"""
        with self.client.get("/prediction-history?limit=10", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
                print(f"Prediction history successful: {response.status_code}")
            else:
                response.failure(f"Prediction history failed: {response.status_code}")
                print(f"Prediction history failed: {response.status_code}")

# Stress test class
class StressTestUser(HttpUser):
    wait_time = between(0.1, 0.5)
    weight = 2
    
    @task
    def rapid_health_checks(self):
        """Rapid health check requests"""
        response = self.client.get("/health")
        if response.status_code == 200:
            print(f"Rapid health check: {response.status_code}")
        else:
            print(f"Rapid health check failed: {response.status_code}")
