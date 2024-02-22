**Warning:**
This project is currently under development and is not ready for production use. Please be aware that schema changes may occur, which could potentially break the application until version 1.0 is released. Use with caution.


# OpenRSVP

OpenRSVP is an open-source event management application built with FastAPI. It allows users to create events, manage RSVPs, and receive notifications for upcoming events.

## Features

- Create new events with a unique secret link.
- Manage RSVPs with options for Yes, No, and Maybe.
- Modify RSVPs through links sent via email.
- Subscribe to email notifications for events.

## Getting Started

### Prerequisites

- Python 3.x
- [FastAPI](https://fastapi.tiangolo.com/)
- [Jinja2](https://jinja.palletsprojects.com/)

### Installation

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Run the FastAPI application:

    ```bash
    uvicorn main:app --reload
    ```

2. Open your browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to access the OpenRSVP application.

## Project Structure

- `main.py`: FastAPI application code.
- `templates/`: HTML templates.
- `static/`: Static files (CSS, JS, etc.).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/): A modern, fast web framework for building APIs with Python.
- [Jinja2](https://jinja.palletsprojects.com/): A template engine for Python.
