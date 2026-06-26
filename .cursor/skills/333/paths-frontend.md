# Frontend scope for /333

See full list: [../uploadm8-frontend-push/paths.md](../uploadm8-frontend-push/paths.md)

**Robocopy (PowerShell):**

```powershell
$src = "C:\Users\Earl\Dev\uploadm8-auth\frontend"
$dst = "C:\Users\Earl\Dev\uploadm8-frontend"
robocopy $src $dst /E /XD node_modules .git /XF .env .env.* *.pem *-credentials.json credentials.json desktop.ini /NFL /NDL /NJH /NJS /nc /ns /np
```

**Remote:** `https://github.com/sosswced/uploadm8-frontend`
