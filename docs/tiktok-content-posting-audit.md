# TikTok Content Posting API — Direct Post (audited)

UploadM8 publishes via TikTok **Content Posting API Direct Post**
(`POST /v2/post/publish/video/init/` + `FILE_UPLOAD` chunks). This is not Share Kit
or inbox-draft upload mode.

## Production enablement (audit approved)

Audited Direct Post is the **default** (unset / `TIKTOK_APP_AUDITED=1`). Pin on
**both** API and worker for clarity:

| Variable | Value | Effect |
|----------|-------|--------|
| `TIKTOK_APP_AUDITED` | `1` (default) | Public Direct Post — honors user privacy (Everyone / Friends / Followers / Only me) |
| `TIKTOK_FORCE_PRIVATE_UNAUDITED` | unset | Do not set unless rolling back to private-only |

Set `TIKTOK_APP_AUDITED=0` only to revert to private-only UX (worker clamps to
`SELF_ONLY` and Upload disables other visibility options).

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

Requires `META_APP_SECRET` set in production. After deploy, reconnect Facebook/Instagram once so tokens store the Meta user ASID for reliable callback matching.

## Smoke test after enabling audited mode

1. Sign in → Upload → select short video → TikTok account
2. Confirm green **Direct Post enabled** banner (not the yellow audit lock banner)
3. Select **Everyone** (or Friends/Followers) — option must be enabled
4. Consent + Upload & Publish
5. On TikTok profile, confirm post is public (not Only me / Inbox-only)

## Rollback

```
TIKTOK_FORCE_PRIVATE_UNAUDITED=1
```

Restart API + worker. Publish clamps to ``SELF_ONLY`` only when this force flag is set.
`TIKTOK_APP_AUDITED=0` affects UI labeling only (does not rewrite privacy at publish).

## Reviewer / compliance notes

- Privacy dropdown must not pre-select Public (except unaudited Only-me default)
- creator_info before export UI
- Music Usage Confirmation + consent before publish
- No silent privacy override without in-app notice when clamped
