# AutoSight frontend

React/Vite client for the FastAPI backend.

## Local development

```powershell
npm install
Copy-Item .env.example .env
npm run dev
```

Set `VITE_API_URL` to the Hugging Face Space URL in Vercel, for example:

```text
VITE_API_URL=https://your-space-name.hf.space
```

For Vercel, set the project root directory to `frontend`, framework preset to `Vite`, and build command to `npm run build`.
