FROM python:3.8

# Step 2: Add requirements.txt file 
COPY src/requirements.txt .

# Step 3: Install required pyhton dependencies from requirements file
RUN pip install -r requirements.txt

# Step 4: Copy the trained model and app
COPY data/06_models/bee_health_model data/06_models/bee_health_model
COPY app.py .

# Step 5: Expose the port FastAPI is running on
EXPOSE 7000

# Step 6: Run FastAPI
CMD ["uvicorn", "app:app", "--host=0.0.0.0", "--port=7000"]