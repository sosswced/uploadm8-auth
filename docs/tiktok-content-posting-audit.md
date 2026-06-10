# TikTok Content Posting API — audit submission kit

Use this when resubmitting UploadM8 for TikTok Direct Post audit.

## App description (paste into TikTok Developer Portal)

UploadM8 is a multi-platform video publishing workspace for creators and agencies. Users upload a video once, configure captions and thumbnails, and publish to TikTok, YouTube, Instagram, and Facebook.

For TikTok, UploadM8 implements the required Content Posting export UX: we call `/v2/post/publish/creator_info/query/` before every post, display the creator avatar and username, let the user manually select privacy from `privacy_level_options` (no default), configure Comment/Duet/Stitch (off by default), complete commercial content disclosure when applicable, and confirm consent with TikTok's Music Usage Confirmation before publishing. While our app is pending audit, posts publish as **Only me (private)** and users are informed in-app before they upload.

Live app: https://app.uploadm8.com  
Privacy policy: https://app.uploadm8.com/privacy.html  
Data deletion: https://app.uploadm8.com/data-deletion.html  
Terms: https://app.uploadm8.com/terms.html

## Demo video shot list (2–3 minutes)

1. Sign in at https://app.uploadm8.com/login.html
2. Open **Upload** → select a short test video
3. Check **TikTok** → **Post to TikTok** panel appears
4. Show **audit banner** (private-only until audit passes)
5. Show **creator avatar + @username** loaded from creator_info API
6. **Manually select** visibility (e.g. Everyone) — no pre-selected default
7. Show interaction toggles (Comment/Duet/Stitch) **unchecked by default**
8. Optionally toggle **Disclose commercial content** → select **Your brand** or **Branded content** → show label prompts
9. Check **consent** with Music Usage Confirmation link visible
10. Click **Upload & Publish** → show queue processing
11. On TikTok app/profile, show post as **Only me** (expected while unaudited)
12. Narrate: "User chose visibility X; app posts private until audit; user was notified before upload."

## Environment (production)

| Variable | Before audit | After TikTok approves |
|----------|--------------|------------------------|
| `TIKTOK_APP_AUDITED` | unset or `0` | `1` |

Do **not** set `TIKTOK_APP_AUDITED=1` until TikTok confirms audit pass.

## Reviewer test account

Provide a test login or clear signup path. Reviewers expect:

- Working signup/login
- TikTok OAuth connect on **Connected Accounts**
- Upload flow with Post to TikTok panel
- Public privacy policy and data deletion URLs reachable without login

## Common rejection causes (avoid)

- Privacy dropdown pre-selected to Public
- No creator_info call before export UI
- No consent / Music Usage Confirmation before publish
- Silent override of user privacy without in-app notice
- Publishing to TikTok without export settings (API-only path)
- Demo video skips the Post to TikTok panel
