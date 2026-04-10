from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Literal, Dict
from datetime import datetime


# ============================================================
# USER PREFERENCES SYSTEM
# ============================================================

class PlatformHashtags(BaseModel):
    tiktok: List[str] = Field(default_factory=list)
    youtube: List[str] = Field(default_factory=list)
    instagram: List[str] = Field(default_factory=list)
    facebook: List[str] = Field(default_factory=list)

class UserPreferencesUpdate(BaseModel):
    # Accept both snake_case (backend) and camelCase (frontend) keys.
    auto_captions: bool = Field(False, alias="autoCaptions")
    auto_thumbnails: bool = Field(False, alias="autoThumbnails")
    styled_thumbnails: bool = Field(True, alias="styledThumbnails")
    thumbnail_interval: int = Field(5, ge=1, le=60, alias="thumbnailInterval")

    default_privacy: Literal["public", "private", "unlisted"] = Field("public", alias="defaultPrivacy")

    ai_hashtags_enabled: bool = Field(False, alias="aiHashtagsEnabled")
    ai_hashtag_count: int = Field(5, ge=1, le=30, alias="aiHashtagCount")
    ai_hashtag_style: Literal["lowercase", "capitalized", "camelcase", "mixed"] = Field("mixed", alias="aiHashtagStyle")
    hashtag_position: Literal["start", "end", "caption"] = Field("end", alias="hashtagPosition")

    max_hashtags: int = Field(15, ge=1, le=50, alias="maxHashtags")
    always_hashtags: List[str] = Field(default_factory=list, alias="alwaysHashtags")
    blocked_hashtags: List[str] = Field(default_factory=list, alias="blockedHashtags")
    platform_hashtags: PlatformHashtags = Field(default_factory=PlatformHashtags, alias="platformHashtags")
    email_notifications: bool = Field(True, alias="emailNotifications")
    discord_webhook: Optional[str] = Field(None, alias="discordWebhook")
    # Caption & AI (stored in users.preferences; worker caption_stage reads these)
    caption_style: Literal["story", "punchy", "factual"] = Field("story", alias="captionStyle")
    caption_tone: Literal["hype", "calm", "cinematic", "authentic"] = Field("authentic", alias="captionTone")
    caption_voice: Literal["default", "mentor", "hypebeast", "best_friend", "teacher", "cinematic_narrator"] = Field("default", alias="captionVoice")
    caption_frame_count: int = Field(6, ge=2, le=12, alias="captionFrameCount")

    class Config:
        populate_by_name = True
        extra = "ignore"


# ============================================================
# Pydantic Models
# ============================================================
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str = Field(min_length=2)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class RefreshRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=16)
    new_password: str = Field(min_length=8)


class UploadInit(BaseModel):
    filename: str
    file_size: int
    content_type: str
    platforms: List[str]
    target_accounts: List[str] = []  # platform_tokens.id UUIDs — publish to specific accounts
    title: str = ""
    caption: str = ""
    hashtags: List[str] = []
    privacy: str = "public"
    scheduled_time: Optional[datetime] = None
    schedule_mode: str = "immediate"  # immediate | scheduled | smart
    has_telemetry: bool = False
    use_ai: bool = False
    smart_schedule_days: int = 7  # How many days to spread uploads across

class SettingsUpdate(BaseModel):
    discord_webhook: Optional[str] = Field(None, alias="discordWebhook")
    telemetry_enabled: Optional[bool] = Field(None, alias="telemetryEnabled")
    hud_enabled: Optional[bool] = Field(None, alias="hudEnabled")
    hud_position: Optional[str] = Field(None, alias="hudPosition")
    speeding_mph: Optional[int] = Field(None, alias="speedingMph")
    euphoria_mph: Optional[int] = Field(None, alias="euphoriaMph")
    hud_speed_unit: Optional[str] = None
    hud_color: Optional[str] = None
    hud_font_family: Optional[str] = None
    hud_font_size: Optional[int] = None
    ffmpeg_screenshot_interval: Optional[int] = None
    auto_generate_thumbnails: Optional[bool] = None
    auto_generate_captions: Optional[bool] = None
    auto_generate_hashtags: Optional[bool] = None
    default_hashtag_count: Optional[int] = None
    always_use_hashtags: Optional[bool] = None

    class Config:
        populate_by_name = True

class CheckoutRequest(BaseModel):
    lookup_key: str
    kind: str = "subscription"  # subscription | topup | addon

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8)

class ProfileUpdateSettings(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    timezone: Optional[str] = None

class PreferencesUpdate(BaseModel):
    """Settings page / legacy prefs — includes Caption & AI card fields for save/load."""
    emailNotifs: Optional[bool] = None
    uploadCompleteNotifs: Optional[bool] = None
    marketingEmails: Optional[bool] = None
    theme: Optional[str] = None
    accentColor: Optional[str] = None
    defaultPrivacy: Optional[str] = None
    autoPublish: Optional[bool] = None
    alwaysHashtags: Optional[List[str]] = None
    blockedHashtags: Optional[List[str]] = None
    tiktokHashtags: Optional[str] = None
    youtubeHashtags: Optional[str] = None
    instagramHashtags: Optional[str] = None
    facebookHashtags: Optional[str] = None
    hashtagPosition: Optional[str] = None
    maxHashtags: Optional[int] = None
    aiHashtagsEnabled: Optional[bool] = None
    aiHashtagCount: Optional[int] = None
    aiHashtagStyle: Optional[str] = None  # lowercase | capitalized | camelcase | mixed
    captionStyle: Optional[str] = None   # story | punchy | factual
    captionTone: Optional[str] = None    # hype | calm | cinematic | authentic
    captionVoice: Optional[str] = None   # default | mentor | hypebeast | best_friend | teacher | cinematic_narrator
    platformHashtags: Optional[dict] = None

class TransferRequest(BaseModel):
    from_platform: str
    to_platform: str
    amount: int

class AnnouncementRequest(BaseModel):
    title: str
    body: str
    send_email: bool = True
    send_discord_community: bool = True
    send_user_webhooks: bool = False
    target: str = "all"  # all | paid | trial | free | specific_tiers
    target_tiers: List[str] = []

class AdminUserUpdate(BaseModel):
    subscription_tier: Optional[str] = None
    role: Optional[str] = None
    status: Optional[str] = None
    flex_enabled: Optional[bool] = None


class AdminWalletAdjust(BaseModel):
    wallet: str  = Field(..., pattern="^(put|aic)$")
    mode:   str  = Field(..., pattern="^(add|subtract|set)$")
    amount: int  = Field(..., ge=0, le=999999)
    reason: str  = Field(..., min_length=3, max_length=200)



class AdminUpdateEmailIn(BaseModel):
    email: EmailStr

class AdminResetPasswordIn(BaseModel):
    temp_password: str = Field(min_length=8, max_length=128)

class SmartScheduleOnlyUpdate(BaseModel):
    """PATCH /api/scheduled/{id} - only smart_schedule (platform -> ISO datetime string)."""
    smart_schedule: Dict[str, str] = Field(..., description="Platform -> ISO datetime string")

class UploadUpdate(BaseModel):
    """PATCH /api/uploads/{id} - title, caption, hashtags, scheduled_time, smart_schedule."""
    title: Optional[str] = None
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None
    scheduled_time: Optional[datetime] = None
    smart_schedule: Optional[Dict[str, str]] = Field(None, description="Platform -> ISO datetime string")

class ColorPreferencesUpdate(BaseModel):
    tiktok_color: Optional[str] = None
    youtube_color: Optional[str] = None
    instagram_color: Optional[str] = None
    facebook_color: Optional[str] = None
    accent_color: Optional[str] = None


# ============================================================
# Profile
# ============================================================

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    timezone: Optional[str] = None


# ============================================================
# Upload completion
# ============================================================

class CompleteUploadBody(BaseModel):
    """Optional metadata from upload page (single-file manual title/caption/hashtags)."""
    title: Optional[str] = None
    caption: Optional[str] = None
    hashtags: Optional[List[str]] = None
    platforms: Optional[List[str]] = None
    privacy: Optional[str] = None
    target_accounts: Optional[List[str]] = None
    group_ids: Optional[List[str]] = None


# ============================================================
# Account Groups
# ============================================================

class AccountGroupIn(BaseModel):
    name: str
    color: str | None = None
    account_ids: list[str] | None = None

class AccountGroupUpdate(BaseModel):
    name: str | None = None
    color: str | None = None
    account_ids: list[str] | None = None


class GroupUpsert(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    account_ids: Optional[List[str]] = None  # platform_tokens.id values as strings
    members: Optional[List[str]] = None      # frontend alias for account_ids


# ============================================================
# Support
# ============================================================

class SupportContactRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    subject: str
    message: str


# ============================================================
# Activity Log
# ============================================================

class ActivityLogIn(BaseModel):
    action: str
    event_category: str = "UI_ACTION"
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[dict] = None
    session_id: Optional[str] = None


# ============================================================
# Discord Notification Settings
# ============================================================

class NotificationSettings(BaseModel):
    notify_mrr_charge: bool = False
    notify_topup: bool = False
    notify_upgrade: bool = False
    notify_downgrade: bool = False
    notify_cancel: bool = False
    notify_refund: bool = False
    notify_openai_cost: bool = False
    notify_storage_cost: bool = False
    notify_compute_cost: bool = False
    notify_weekly_report: bool = False
    notify_stripe_payout: bool = False
    notify_cloud_billing: bool = False
    notify_render_renewal: bool = False
    stripe_payout_day: int = 15
    cloud_billing_day: int = 1
    render_renewal_day: int = 7
    admin_webhook_url: str = ""
