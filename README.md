# Varnika GlowBot

Varnika GlowBot is a Streamlit MVP for visual skincare journaling and cautious wellness information. It uses NVIDIA open models through Nebius Token Factory in a sequential, two-stage workflow.

## Features

- Capture a photo with the browser camera or upload a PNG, JPEG, or WebP image.
- Stage 1 uses `meta-llama/Llama-3.2-11b-Vision-Instruct` to describe visible appearance and return rough, non-diagnostic 0-4 estimates for pigmentation appearance, acne-like appearance, visible scarring, and fine-line appearance.
- Stage 2 uses `nvidia/Llama-3.1-Nemotron-70B-Instruct` to evaluate Stage 1's structured output and provide cautious topical-care, balanced food-habit, general-wellness, and gentle face-yoga suggestions.
- Ask follow-up questions about the latest scan and recommendations in a session-only chat.
- Keep timestamped analysis records in a local SQLite database and view score estimates over time.
- Delete all local records for a profile from the history tab. The app does not save the uploaded image or an image fingerprint; it saves estimates, observations, and generated guidance.

## Setup

1. Install Python 3.10 or later.
2. From the repository root, create and activate a virtual environment:

	```bash
	python -m venv .venv
	source .venv/bin/activate
	```

	On Windows PowerShell, activate it with `.venv\\Scripts\\Activate.ps1`.

3. Install dependencies:

	```bash
	pip install -r requirements.txt
	```

4. In the Nebius Token Factory console, create an API key and confirm that the selected models are available to your account.
5. Put the key in the root `.env` file, replacing the placeholder:

	```dotenv
	NEBIUS_API_KEY="your-real-token-factory-key"
	```

	Keep `.env` private. It is excluded by `.gitignore`. For deployments, inject the key through your hosting platform's secret or environment-variable settings instead of committing it.

6. Launch the app:

	```bash
	streamlit run app.py
	```

7. Open the local URL printed by Streamlit, usually `http://localhost:8501`. Allow camera access when prompted, or choose the upload option.
8. Enter a pseudonymous profile ID, such as `my-profile`; do not use a name, email, or other identifying information.
9. Choose a photo and focus, confirm you are at least 18 and have permission to submit the photo, then confirm that it will be sent to Nebius for inference. Run the analysis.
10. Use **Progress history** to review previous estimates. For more comparable photos, use the same camera, distance, lighting, and expression. Use the delete control to remove the selected profile's local records.

## Configuration

- Nebius Token Factory's OpenAI-compatible endpoint is `https://api.tokenfactory.nebius.com/v1/`.
- Set `GLOWBOT_DB_PATH` to change the SQLite database location. By default, `glowbot_records.sqlite3` is created in the repository root and is ignored by Git.
- A local SQLite file is suitable for a single-user demo. Use authenticated, managed storage and access controls before deploying for multiple users.

## Safety and limitations

This hackathon prototype is not a medical device, diagnostic tool, or treatment recommendation. Scores are subjective model estimates and can vary with image quality, lighting, camera processing, skin tone, and angle; a score change does not establish clinical improvement. AI-generated remedies and face-yoga suggestions may be incomplete or unsuitable for an individual. Do not use DIY irritants or ingest topical ingredients; seek a licensed clinician for persistent, painful, rapidly changing, or concerning skin findings.

Photos are sent to the configured hosted model service for inference. The application does not persist image bytes, but does persist the model's observations and response locally. Avoid submitting images or details you do not want processed or retained. The app requires user confirmation of adulthood and photo permission, but this checkbox is not identity verification or a substitute for deployment-level consent and privacy controls.
