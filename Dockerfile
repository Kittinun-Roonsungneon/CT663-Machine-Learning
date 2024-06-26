# Use an official Python runtime as a parent image
FROM python:3.11.2-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy the Linux-specific requirements file
COPY requirements-linux.txt /app/requirements.txt

# Install any needed packages specified in requirements-linux.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8888 for Jupyter Lab
EXPOSE 8888

# Run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
