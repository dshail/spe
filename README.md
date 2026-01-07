# AI Exam Auto-Grader

## Deployment on Streamlit Cloud

### Prerequisites
- A GitHub account.
- This repository pushed to GitHub.

### Steps to Deploy
1. Log in to [Streamlit Cloud](https://streamlit.io/cloud).
2. Click **New app**.
3. Select your repository, branch, and main file (`app.py`).
4. Click **Advanced settings** (or do this after deployment in the App Settings).

### Securing API Keys
To keep your API keys secure, **DO NOT** commit them to GitHub. Instead, use Streamlit Secrets:

1. In your Streamlit Cloud dashboard, go to your app.
2. Click the **Settings** (three dots) -> **Secrets**.
3. Paste the following configuration:

```toml
DATALAB_API_KEY = "your-datalab-api-key-here"
GEMINI_API_KEY = "your-gemini-api-key-here"
```

The application is configured to automatically look for these secrets. If they are found, they will pre-fill the fields (or handle authentication silently). If not found, you can still manually enter them in the sidebar.

### Local Development
To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```
You can also use a `.streamlit/secrets.toml` file locally for convenience.
