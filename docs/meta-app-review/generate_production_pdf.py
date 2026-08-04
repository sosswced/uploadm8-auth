#!/usr/bin/env python3
"""Generate UploadM8 Meta App Review production set PDF."""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).with_name("UploadM8_Meta_App_Review_Production_Set.pdf")


class Doc(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"UploadM8 Meta App Review Production Set  |  Page {self.page_no()}/{{nb}}", align="C")

    def _reset_x(self):
        self.set_x(self.l_margin)

    def h1(self, text):
        self._reset_x()
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 8, text)
        self.ln(2)
        self._reset_x()

    def h2(self, text):
        self.ln(2)
        self._reset_x()
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 7, text)
        self.ln(1)
        self._reset_x()

    def h3(self, text):
        self.ln(1)
        self._reset_x()
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(0.5)
        self._reset_x()

    def body(self, text):
        self._reset_x()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(25, 25, 25)
        self.multi_cell(0, 5, text)
        self.ln(1)
        self._reset_x()

    def bullet(self, text):
        self._reset_x()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(25, 25, 25)
        self.multi_cell(0, 5, f"- {text}")
        self._reset_x()

    def callout(self, title, text):
        self._reset_x()
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(120, 70, 0)
        self.multi_cell(0, 5, title)
        self._reset_x()
        self.set_font("Helvetica", "", 9)
        self.set_text_color(25, 25, 25)
        self.multi_cell(0, 5, text)
        self.ln(2)
        self._reset_x()

    def paste(self, text):
        self._reset_x()
        self.set_fill_color(245, 245, 245)
        self.set_font("Courier", "", 8)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 4.2, text, fill=True)
        self.ln(2)
        self._reset_x()

    def step(self, n, action, say="", hold="", caption=""):
        self._reset_x()
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(0, 80, 140)
        self.multi_cell(0, 5, f"STEP {n}: {action}")
        self._reset_x()
        self.set_font("Helvetica", "", 8)
        self.set_text_color(40, 40, 40)
        if say:
            self.multi_cell(0, 4.5, f"  DO: {say}")
            self._reset_x()
        if hold:
            self.multi_cell(0, 4.5, f"  HOLD: {hold}")
            self._reset_x()
        if caption:
            self.multi_cell(0, 4.5, f"  ON-SCREEN CAPTION: {caption}")
            self._reset_x()
        self.ln(1)


def build():
    pdf = Doc(format="Letter")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # COVER
    def ctext(size, style, text, h=None):
        pdf.set_x(pdf.l_margin)
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(0, h or max(5, int(size * 0.45)), text, align="C")
        pdf.set_x(pdf.l_margin)

    pdf.ln(20)
    ctext(22, "B", "UploadM8", 10)
    ctext(14, "B", "Meta App Review - Full Production Set", 8)
    pdf.ln(4)
    ctext(
        10,
        "",
        "Move scripts, paste-ready Policy 1.6 answers, screencast shot list,\n"
        "and resubmission checklist for the denied permissions.",
        5,
    )
    pdf.ln(8)
    ctext(10, "B", "DENIED (resubmit with revised notes + screencasts)", 6)
    ctext(
        9,
        "",
        "pages_read_user_content  |  publish_video (DROP)  |  instagram_content_publish\n"
        "read_insights  |  instagram_manage_insights  |  instagram_basic",
        5,
    )
    pdf.ln(4)
    ctext(10, "B", "ALREADY APPROVED (keep; do not re-argue)", 6)
    ctext(9, "", "pages_show_list  |  business_management  |  pages_read_engagement", 5)
    pdf.ln(6)
    ctext(10, "B", "NEW PERMISSION TO ADD", 6)
    ctext(
        9,
        "",
        "pages_manage_posts - required for Page VOD / Reels publish via POST /{page-id}/videos\n"
        "(replace publish_video; Meta defines publish_video as LIVE streaming only)",
        5,
    )

    # SECTION 1
    pdf.add_page()
    pdf.h1("1. Why you were rejected (Policy 1.6)")
    pdf.body(
        "Every denied permission used the same rejection: Developer Policy 1.6 - "
        "use case invalid or not needed for core functionality. That means reviewers "
        "did not believe (from your notes + screencasts) that each permission is required "
        "for a live, user-visible UploadM8 feature."
    )
    pdf.h3("Root causes in your last submission")
    pdf.bullet(
        "publish_video: wrong permission. Meta allowed usage = LIVE video streaming. "
        "UploadM8 publishes recorded Page videos/Reels. That is pages_manage_posts."
    )
    pdf.bullet(
        "Insights / instagram_basic screencasts were ~0:31 - too short to show login, "
        "grant, feature, and result. Reviewers need end-to-end proof."
    )
    pdf.bullet(
        "Notes leaned on API jargon. Meta asks three product questions: (1) which UI "
        "needs it, (2) how integration works, (3) how it helps the end user."
    )
    pdf.bullet(
        "pages_read_user_content was framed only as analytics plumbing. Reviewers need "
        "to SEE Page videos listed in the product UI, then metrics appear."
    )
    pdf.callout(
        "HARD RULE FOR RESUBMIT",
        "DROP publish_video from this submission. ADD pages_manage_posts. "
        "Record 2.5-6 minutes per permission (or clearly labeled sections). "
        "Never cut from dashboard mid-flow - always show Meta Login + grant + feature + proof.",
    )

    # SECTION 2
    pdf.h1("2. What to request this round")
    pdf.h3("Request Advanced Access for")
    pdf.bullet("instagram_basic")
    pdf.bullet("instagram_content_publish")
    pdf.bullet("instagram_manage_insights")
    pdf.bullet("pages_manage_posts   <- NEW (Facebook publish)")
    pdf.bullet("pages_read_user_content")
    pdf.bullet("read_insights")
    pdf.h3("Do NOT request")
    pdf.bullet("publish_video (live-only; will fail Policy 1.6 again)")
    pdf.h3("Already approved - leave alone")
    pdf.bullet("pages_show_list, business_management, pages_read_engagement")

    pdf.h2("Dependency map (must show in videos)")
    pdf.bullet("instagram_content_publish needs: instagram_basic + pages_read_engagement + pages_show_list")
    pdf.bullet("instagram_manage_insights needs: instagram_basic + pages_read_engagement + pages_show_list")
    pdf.bullet("pages_manage_posts needs: pages_read_engagement + pages_show_list")
    pdf.bullet("read_insights needs: pages_read_engagement + pages_show_list")
    pdf.bullet("pages_read_user_content needs: pages_show_list")

    # SECTION 3
    pdf.add_page()
    pdf.h1("3. Production set (day-of prep)")
    pdf.h2("A. Environment")
    pdf.bullet("Production app URL reachable without VPN.")
    pdf.bullet("META_OAUTH_MODE=full (must request denied scopes on consent screen).")
    pdf.bullet("App in Live or Development with reviewer test user added as Tester/Developer.")
    pdf.bullet("Make 1 successful Graph API call per permission within 30 days before submit.")
    pdf.bullet("English UI. Zoom browser to 110-125% so text is readable at 1080p.")

    pdf.h2("B. Test accounts (give Meta these in reviewer notes)")
    pdf.bullet("UploadM8 login email + password (dedicated reviewer account).")
    pdf.bullet("Facebook user who admins ONE test Page (CREATE_CONTENT task).")
    pdf.bullet("Instagram Professional (Business or Creator) linked to that Page.")
    pdf.bullet("Page + IG already have at least 1 published video/Reel so insights are non-empty.")
    pdf.bullet("Disconnect FB/IG from UploadM8 before recording so OAuth is shown fresh.")

    pdf.h2("C. Props / assets")
    pdf.bullet("Short vertical MP4 (~15-30s), under platform size limits, non-copyrighted.")
    pdf.bullet("Caption text prepared (no spammy hashtag walls).")
    pdf.bullet("Screen recorder: OBS / Camtasia / Snagit / Xbox Game Bar - 1080p+, 30fps.")
    pdf.bullet("Optional: large on-screen text overlay tool (or edit captions in post).")

    pdf.h2("D. Files to produce (upload one screencast per permission)")
    pdf.bullet("SC01_instagram_basic.mp4                 (2:30-4:00)")
    pdf.bullet("SC02_instagram_content_publish.mp4       (4:00-6:00)")
    pdf.bullet("SC03_pages_manage_posts.mp4              (4:00-6:00)")
    pdf.bullet("SC04_pages_read_user_content.mp4         (3:00-5:00)")
    pdf.bullet("SC05_read_insights.mp4                   (3:00-5:00)")
    pdf.bullet("SC06_instagram_manage_insights.mp4       (3:00-5:00)")
    pdf.body(
        "You may film one long master take, then cut into SC01-SC06. "
        "Each uploaded file must still start with Login + grant for that permission."
    )

    pdf.h2("E. Caption style (every video)")
    pdf.bullet("Bottom third, white text, dark bar, 2 lines max.")
    pdf.bullet("Always name the permission in CAPS when it is used.")
    pdf.bullet("Example: 'INSTAGRAM_BASIC - loading IG username into Connected Accounts'")

    # SECTION 4 move scripts
    pdf.add_page()
    pdf.h1("4. Full move scripts")

    pdf.h2("SC01 - instagram_basic (connect + profile)")
    pdf.body("Goal: prove UploadM8 needs IG profile identity in the product UI.")
    pdf.step(1, "Cold start", "Open uploadm8.com -> Login with reviewer account.", "2s on dashboard", "UploadM8 reviewer login")
    pdf.step(2, "Open Platforms", "Go to Platforms / Connected Accounts.", "2s", "Connected Accounts")
    pdf.step(3, "Start Instagram connect", "Click Connect Instagram. Meta Login appears.", "1s", "Starting Meta Login for Instagram")
    pdf.step(
        4,
        "Show consent scopes",
        "Slowly scroll/highlight scopes. Point mouse at instagram_basic.",
        "4-6s freeze on permission list",
        "GRANTING: instagram_basic",
    )
    pdf.step(5, "Approve + select Page/IG", "Accept. Select the Business Page / IG account if prompted.", "2s", "Selecting Instagram Professional account")
    pdf.step(
        6,
        "Proof in product",
        "Back on Platforms: show IG username + avatar connected.",
        "5s hold",
        "INSTAGRAM_BASIC RESULT - username/avatar shown in UploadM8",
    )
    pdf.step(7, "Optional media list bridge", "Open Analytics briefly and show IG account context loading.", "3s", "INSTAGRAM_BASIC enables media list for analytics")

    pdf.h2("SC02 - instagram_content_publish (end-to-end Reel publish)")
    pdf.body("Goal: create -> publish -> live Instagram Reel. Do NOT skip live proof.")
    pdf.step(1, "Login + Platforms", "Show already-connected IG or reconnect showing grant.", "3s", "INSTAGRAM_CONTENT_PUBLISH on consent")
    pdf.step(2, "Upload flow", "Open Upload. Select vertical video. Select Instagram destination only (or clearly checked).", "3s", "User selects Instagram destination")
    pdf.step(3, "Caption", "Enter caption. Show privacy if present.", "2s", "User-authored caption - no silent publish")
    pdf.step(4, "Publish click", "Click Publish / Upload. Hold on confirmation.", "3s", "INSTAGRAM_CONTENT_PUBLISH - create + publish media")
    pdf.step(5, "Queue success", "Open Queue / upload detail. Show success + Instagram link.", "4s", "Publish succeeded - platform URL present")
    pdf.step(6, "Live proof", "Open the Instagram post/Reel in browser (or Instagram). Show the live video.", "6-8s", "LIVE PROOF - Reel on Instagram Business account")

    pdf.add_page()
    pdf.h2("SC03 - pages_manage_posts (Facebook Page video publish)  [REPLACES publish_video]")
    pdf.body(
        "Goal: publish organic video to a Facebook Page. Say aloud/caption that this is "
        "pages_manage_posts for Page video posts - NOT live streaming."
    )
    pdf.step(1, "Connect Facebook", "Platforms -> Connect Facebook. Show consent including pages_manage_posts.", "5s", "GRANTING: pages_manage_posts")
    pdf.step(2, "Select Page", "Choose the test Page. Confirm it appears connected.", "3s", "Page selected for publishing")
    pdf.step(3, "Upload", "Upload -> select Facebook destination -> caption -> Publish.", "3s", "User intent: publish video to Page")
    pdf.step(4, "Processing", "Show queue processing / success state.", "4s", "POST /{page-id}/videos via pages_manage_posts")
    pdf.step(5, "Live proof", "Open facebook.com Page -> Videos / Reels. Show the new video.", "8s", "LIVE PROOF - video on Facebook Page")

    pdf.h2("SC04 - pages_read_user_content (list Page videos in Analytics)")
    pdf.body(
        "Goal: show Page video inventory in UploadM8. This is the permission Meta said "
        "was 'not needed' - you must make the UI dependency obvious."
    )
    pdf.step(1, "Consent reminder", "If reconnecting, highlight pages_read_user_content on consent.", "4s", "GRANTING: pages_read_user_content")
    pdf.step(2, "Open Analytics", "Analytics -> Platform Stats -> Facebook.", "2s", "Analytics - Facebook Page content")
    pdf.step(3, "Trigger list", "Refresh / sync stats. Wait for videos to load.", "4s", "GET /{page-id}/videos requires pages_read_user_content")
    pdf.step(4, "Show list", "Point to listed Page videos/Reels (titles/thumbnails/IDs).", "6s", "RESULT - Page videos listed for the Page owner")
    pdf.step(5, "Why it matters", "Click one video row / detail if available; show it feeds analytics.", "4s", "Listing enables per-video performance for the creator")

    pdf.h2("SC05 - read_insights (Facebook video metrics)")
    pdf.step(1, "Consent", "Highlight read_insights on grant screen if reconnecting.", "4s", "GRANTING: read_insights")
    pdf.step(2, "Analytics FB card", "Open Analytics Facebook section with non-zero metrics.", "3s", "Facebook insights for Page owner only")
    pdf.step(3, "Show views", "Point to video views / reactions / comments / shares.", "6s", "read_insights - total_video_views + engagement")
    pdf.step(4, "End-user value", "Hold on cross-platform comparison if visible.", "4s", "Creator sees which FB videos perform best")

    pdf.h2("SC06 - instagram_manage_insights (IG Reel metrics)")
    pdf.step(1, "Consent", "Highlight instagram_manage_insights.", "4s", "GRANTING: instagram_manage_insights")
    pdf.step(2, "Analytics IG", "Analytics -> Platform Stats -> Instagram.", "2s", "Instagram insights dashboard")
    pdf.step(3, "Advanced metrics", "Point to plays, reach, saves, shares (not just likes).", "6s", "instagram_manage_insights - plays/reach/saved/shares")
    pdf.step(4, "Owner-only", "Show this is the logged-in user's account only.", "3s", "Shown only to authenticated account owner")

    # SECTION 5 paste text
    pdf.add_page()
    pdf.h1("5. Paste-ready Meta form answers (Policy 1.6)")
    pdf.body(
        "Paste each block into 'Tell us how you're using this permission'. "
        "Each block answers Meta's three required items."
    )

    pdf.h3("instagram_basic")
    pdf.paste(
        "1) Which functionality requires this permission:\n"
        "UploadM8 Connected Accounts and Analytics. After a creator connects Instagram,\n"
        "we display the Instagram Professional account username/profile identity on the\n"
        "Platforms page and use the Instagram user ID to load that account's media list\n"
        "for Analytics.\n\n"
        "2) How the integration works:\n"
        "During Facebook Login for Business, the user grants instagram_basic. UploadM8\n"
        "resolves the Instagram Business/Creator account linked to their Page, stores the\n"
        "IG user ID, and later calls GET /{ig-user-id}/media to retrieve media IDs owned\n"
        "by that account. Profile fields are used only to label the connected account in UI.\n\n"
        "3) How this enhances the end-user experience:\n"
        "Creators can confirm the correct Instagram account is connected and see analytics\n"
        "for their own media. Without instagram_basic, UploadM8 cannot identify the account\n"
        "or list media needed for publishing destination selection and insights."
    )

    pdf.h3("instagram_content_publish")
    pdf.paste(
        "1) Which functionality requires this permission:\n"
        "UploadM8's core Upload workflow - publishing a user-selected video as an organic\n"
        "Instagram Reel to the creator's own Instagram Professional account.\n\n"
        "2) How the integration works:\n"
        "When the user clicks Publish with Instagram selected, UploadM8 creates a media\n"
        "container (POST /{ig-user-id}/media with video URL + caption), waits until status\n"
        "is FINISHED, then calls POST /{ig-user-id}/media_publish. No post is created without\n"
        "explicit user action. Content is published only to the Instagram account the user\n"
        "connected.\n\n"
        "3) How this enhances the end-user experience:\n"
        "Creators upload once in UploadM8 and publish organic Reels to Instagram without\n"
        "leaving the dashboard, alongside other platforms they manage. This is core product\n"
        "functionality for a multi-platform publisher."
    )

    pdf.add_page()
    pdf.h3("pages_manage_posts  (use this instead of publish_video)")
    pdf.paste(
        "1) Which functionality requires this permission:\n"
        "UploadM8 Facebook Page publishing - uploading a user-selected video to the\n"
        "Facebook Page the user administers (organic Page video / Reels publish).\n\n"
        "2) How the integration works:\n"
        "After the user connects a Page (pages_show_list) and clicks Publish with Facebook\n"
        "selected, UploadM8 uses the Page access token to call POST /{page-id}/videos with\n"
        "the video and user-authored description. Meta's Page Videos API documents\n"
        "pages_manage_posts (with pages_show_list and pages_read_engagement) for this\n"
        "organic publish flow. UploadM8 does not use live streaming.\n\n"
        "3) How this enhances the end-user experience:\n"
        "Page admins can publish organic videos to their Page from UploadM8 with one\n"
        "explicit Publish action, then receive a link to the live Page post in Queue."
    )

    pdf.h3("publish_video - DO NOT RESUBMIT")
    pdf.paste(
        "REMOVE from App Review request.\n"
        "Reason: Meta allowed usage is live-video streaming to timeline/group/event/Page.\n"
        "UploadM8 publishes recorded Page videos via POST /{page-id}/videos, which requires\n"
        "pages_manage_posts, not publish_video. Resubmitting publish_video for VOD will be\n"
        "rejected again under Policy 1.6 (invalid use case)."
    )

    pdf.h3("pages_read_user_content")
    pdf.paste(
        "1) Which functionality requires this permission:\n"
        "UploadM8 Analytics -> Platform Stats for Facebook. The product lists videos that\n"
        "exist on the user's Page so the Page owner can review content performance in one place.\n\n"
        "2) How the integration works:\n"
        "After the Page owner opens Analytics, UploadM8 calls GET /{page-id}/videos to retrieve\n"
        "recent Page videos (IDs/metadata). Those IDs are required to load per-video metrics\n"
        "and present a Facebook content list inside UploadM8. This is read-only for the Page\n"
        "the user administers. We do not scrape other users' profiles or redistribute content.\n\n"
        "3) How this enhances the end-user experience:\n"
        "Creators see which of their Page videos are available for analytics without leaving\n"
        "UploadM8. Without pages_read_user_content, Page video listing is incomplete and the\n"
        "Facebook analytics experience cannot function as a first-class part of the product."
    )

    pdf.h3("read_insights")
    pdf.paste(
        "1) Which functionality requires this permission:\n"
        "UploadM8 Analytics for Facebook Page video performance (views and engagement).\n\n"
        "2) How the integration works:\n"
        "For videos belonging to the connected Page, UploadM8 requests video insights\n"
        "(for example total_video_views and engagement-related insight metrics) and displays\n"
        "them in the authenticated user's Analytics dashboard. Data is shown only to the\n"
        "Page owner inside their account; it is not sold or shared with other customers.\n\n"
        "3) How this enhances the end-user experience:\n"
        "Creators compare Facebook performance with Instagram/TikTok/YouTube in one dashboard\n"
        "and decide what to publish next. Insights are core to UploadM8's analytics value,\n"
        "not an optional side feature."
    )

    pdf.add_page()
    pdf.h3("instagram_manage_insights")
    pdf.paste(
        "1) Which functionality requires this permission:\n"
        "UploadM8 Analytics -> Platform Stats for Instagram Reels/posts owned by the\n"
        "connected Instagram Professional account.\n\n"
        "2) How the integration works:\n"
        "UploadM8 lists the account's media, then calls the Instagram insights edge for\n"
        "metrics such as plays, reach, saved, shares, likes, and comments (using valid\n"
        "metric sets per media type). Results populate the Instagram analytics card for\n"
        "that logged-in owner only.\n\n"
        "3) How this enhances the end-user experience:\n"
        "Creators see Reel performance beyond basic like/comment counts (plays/reach/saves),\n"
        "which helps them improve content strategy inside UploadM8 without opening native\n"
        "Instagram insights separately for every post."
    )

    # SECTION 6 narration
    pdf.h1("6. Spoken narration (optional but recommended)")
    pdf.body("If you record voiceover, keep it slow and literal:")
    pdf.bullet(
        "SC01: 'I am connecting Instagram. I grant instagram_basic. UploadM8 now shows my IG username.'"
    )
    pdf.bullet(
        "SC02: 'I select Instagram, write a caption, click Publish. Here is the live Reel on Instagram.'"
    )
    pdf.bullet(
        "SC03: 'I am publishing a recorded video to my Facebook Page using pages_manage_posts. "
        "This is not a live stream. Here is the video live on the Page.'"
    )
    pdf.bullet(
        "SC04: 'Analytics lists my Page videos. That list requires pages_read_user_content.'"
    )
    pdf.bullet(
        "SC05: 'These Facebook view counts come from read_insights for my Page.'"
    )
    pdf.bullet(
        "SC06: 'These Instagram plays and reach metrics require instagram_manage_insights.'"
    )

    # SECTION 7 checklist
    pdf.h1("7. Resubmission checklist")
    pdf.bullet("[ ] Removed publish_video from permissions request")
    pdf.bullet("[ ] Added pages_manage_posts + screencast SC03")
    pdf.bullet("[ ] Pasted revised notes for all 6 permissions (section 5)")
    pdf.bullet("[ ] Each screencast shows Login -> Grant -> Feature -> Proof")
    pdf.bullet("[ ] Live IG Reel proof in SC02; live FB Page video proof in SC03")
    pdf.bullet("[ ] Analytics videos SC04-SC06 are >= 3 minutes with visible non-zero metrics")
    pdf.bullet("[ ] Reviewer UploadM8 credentials + FB/IG test identity provided")
    pdf.bullet("[ ] Privacy Policy + Data Deletion callback URLs valid")
    pdf.bullet("[ ] Successful API usage logged for each advanced permission")
    pdf.bullet("[ ] META_OAUTH_MODE=full on the environment reviewers will use")

    pdf.h2("Reviewer access note (paste in submission)")
    pdf.paste(
        "Reviewer login:\n"
        "URL: https://uploadm8.com/login.html\n"
        "Email: <REVIEWER_EMAIL>\n"
        "Password: <REVIEWER_PASSWORD>\n\n"
        "Connected test assets (already prepared):\n"
        "- Facebook Page: <PAGE_NAME> (reviewer FB user is admin)\n"
        "- Instagram Professional: <@HANDLE> linked to that Page\n\n"
        "How to verify:\n"
        "1) Platforms - confirm FB Page + IG connected (or Connect and grant requested scopes)\n"
        "2) Upload - publish a short video to Instagram and/or Facebook\n"
        "3) Queue - open live links\n"
        "4) Analytics -> Platform Stats - view Facebook video list + insights and Instagram insights\n\n"
        "Note: Facebook Page publishing uses pages_manage_posts (organic Page video).\n"
        "We are not requesting publish_video (live streaming)."
    )

    pdf.h1("8. Recording day run order (90 minutes)")
    pdf.bullet("00:00 Prep accounts, disconnect Meta from UploadM8, place video file on Desktop")
    pdf.bullet("00:10 Record SC01 (instagram_basic)")
    pdf.bullet("00:25 Record SC02 (instagram publish + live proof)")
    pdf.bullet("00:45 Record SC03 (pages_manage_posts + live proof)")
    pdf.bullet("01:05 Record SC04 + SC05 back-to-back on Analytics Facebook")
    pdf.bullet("01:20 Record SC06 on Analytics Instagram")
    pdf.bullet("01:30 Export MP4s, add captions if missing, upload to Meta form, paste Section 5 text")

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.multi_cell(0, 6, "End of production set. Film the proof. Paste the answers. Drop publish_video.")

    pdf.output(str(OUT))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
