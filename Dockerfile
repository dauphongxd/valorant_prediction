# Use the official Python 3.11 image, which matches your version
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python libraries listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your bot.py and any other project files into the container
COPY . .

# Set the command that will run when the container starts
CMD ["python", "bot.py"]