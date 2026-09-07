# TikTok Content Posting API — Direct Post (audited)

UploadM8 publishes via TikTok **Content Posting API Direct Post**
(`POST /v2/post/publish/video/init/` + `FILE_UPLOAD` chunks). This is not Share Kit
or inbox-draft upload mode (inbox remains an optional “Finish in TikTok app” path).

## Production status

Content Posting API audit is **approved**. Direct Post honors the creator’s selected
privacy level from `privacy_level_options` (Everyone / Friends / Followers / Only me).
There is no private-only clamp and no `TIKTOK_APP_AUDITED` runtime toggle.

OAuth scopes (must match TikTok Developer Portal exactly):

`user.info.basic`, `user.info.stats`, `video.publish`, `video.upload`, `video.list`

- **stats + video.list** → Analytics live cards
- **user.info.profile** → `@username` for watch URLs (add in portal before re-enabling in `OAUTH_CONFIG`)

## App description (Developer Portal)

UploadM8 is a multi-platform video publishing workspace for creators and agencies. Users upload a video once, configure captions and thumbnails, and publish to TikTok, YouTube, Instagram, and Facebook.

For TikTok, UploadM8 implements the required Content Posting export UX: we call `/v2/post/publish/creator_info/query/` before every post, display the creator avatar and username, let the user manually select privacy from `privacy_level_options` (no default), configure Comment/Duet/Stitch (off by default), complete commercial content disclosure when applicable, and confirm consent with TikTok's Music Usage Confirmation before publishing. With Content Posting API audit approved, Direct Post publishes at the visibility the user selects.

Live app: https://app.uploadm8.com  
Privacy policy: https://app.uploadm8.com/privacy.html  
Data deletion: https://app.uploadm8.com/data-deletion.html  
Terms: https://app.uploadm8.com/terms.html  
Refunds: https://app.uploadm8.com/refunds.html  
Support: https://app.uploadm8.com/support.html  

## Meta (Facebook / Instagram) App Dashboard URLs

| Field | URL |
|-------|-----|
| Privacy Policy | https://app.uploadm8.com/privacy.html |
| Terms of Service | https://app.uploadm8.com/terms.html |
| Data Deletion Instructions | https://app.uploadm8.com/data-deletion.html |
| Data Deletion Request Callback | https://auth.uploadm8.com/api/webhooks/facebook/data-deletion |
| Deauthorize Callback | https://auth.uploadm8.com/api/webhooks/facebook/deauthorize |

Requires `META_APP_SECRET` and `META_OAUTH_MODE=full` in production. After deploy,
reconnect Facebook/Instagram once so tokens store the Meta user ASID for deletion
matching. Users who manage multiple Pages (or IG accounts) choose the destination
in the OAuth popup.

## Smoke test

1. Sign in → Upload → select short video → TikTok account
2. Confirm green **Direct Post enabled** banner
3. Select **Everyone** (or Friends/Followers) — option must be enabled
4. Consent + Upload & Publish
5. On TikTok profile, confirm post is public (not Only me / Inbox-only)
6. Analytics → TikTok Live card shows video.list + user.info.stats metrics

## Reviewer / compliance notes

- Privacy dropdown must not pre-select Public
- creator_info before export UI
- Music Usage Confirmation + consent before publish
- Optional Finish in TikTok app → Inbox draft (not Direct Post)
