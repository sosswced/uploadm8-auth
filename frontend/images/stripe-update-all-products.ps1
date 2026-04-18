# =============================================================
#  UploadM8 — Full Product Update Script
#  Updates Stripe products with:
#    • Chargeback-grade descriptions
#    • Statement descriptors
#    • Unit labels
#    • Tax codes (txcd_10103001 = SaaS / digital services)
#    • Full metadata (type, lookup_key, wallet, amount, tier)
#    • Product images (uploads PNG → Stripe Files → attaches)
#
#  Catalog: 4 subscriptions + 5 PUT top-ups + 5 AIC top-ups = 14 products
#
#  Usage:
#    1. Set $ApiKey and $ImageFolder below
#    2. pwsh ./stripe-update-all-products.ps1
#
#  Images expected in $ImageFolder:
#    sub_creator_lite.png   sub_creator_pro.png
#    sub_studio.png         sub_agency.png
#    topup_put_250.png  topup_put_500.png  topup_put_1000.png
#    topup_put_2500.png topup_put_5000.png
#    topup_aic_500.png  topup_aic_1000.png topup_aic_2500.png
#    topup_aic_5000.png topup_aic_10000.png
# =============================================================

$ApiKey      = "sk_test_REPLACE_ME"          # <── your Stripe key
$ImageFolder = "C:\path\to\stripe-images"    # <── folder with PNGs

$Base    = "https://api.stripe.com/v1"
$Headers = @{ Authorization = "Bearer $ApiKey" }

# ── Form-encoded POST helper ──────────────────────────────────
function Invoke-Stripe {
    param([string]$Endpoint, [hashtable]$Body, [string]$Method = "POST")
    $Pairs = @()
    foreach ($kv in $Body.GetEnumerator()) {
        $Pairs += [System.Uri]::EscapeDataString($kv.Key) + "=" + [System.Uri]::EscapeDataString($kv.Value)
    }
    try {
        return Invoke-RestMethod `
            -Uri "$Base/$Endpoint" `
            -Method $Method `
            -Headers $Headers `
            -ContentType "application/x-www-form-urlencoded" `
            -Body ($Pairs -join "&")
    } catch {
        $msg = $_.ErrorDetails.Message | ConvertFrom-Json -ErrorAction SilentlyContinue
        Write-Host "   $Endpoint — $($msg.error.message ?? $_)" -ForegroundColor Red
        return $null
    }
}

# ── Upload PNG → Stripe Files → return file URL ───────────────
function Upload-Image {
    param([string]$FilePath)
    if (-not (Test-Path $FilePath)) {
        Write-Host "     Image not found: $FilePath" -ForegroundColor Yellow
        return $null
    }
    try {
        $fileBytes = [System.IO.File]::ReadAllBytes($FilePath)
        $fileName  = [System.IO.Path]::GetFileName($FilePath)
        $boundary  = [System.Guid]::NewGuid().ToString("N")
        $LF        = "`r`n"

        $bodyLines = @(
            "--$boundary",
            "Content-Disposition: form-data; name=`"purpose`"",
            "",
            "product_image",
            "--$boundary",
            "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"",
            "Content-Type: image/png",
            ""
        )
        $prefix = [System.Text.Encoding]::UTF8.GetBytes(($bodyLines -join $LF) + $LF)
        $suffix = [System.Text.Encoding]::UTF8.GetBytes("$LF--$boundary--$LF")

        $body = New-Object byte[] ($prefix.Length + $fileBytes.Length + $suffix.Length)
        [System.Buffer]::BlockCopy($prefix,    0, $body, 0,                                  $prefix.Length)
        [System.Buffer]::BlockCopy($fileBytes, 0, $body, $prefix.Length,                     $fileBytes.Length)
        [System.Buffer]::BlockCopy($suffix,    0, $body, $prefix.Length + $fileBytes.Length, $suffix.Length)

        $uploadHeaders = @{
            Authorization  = "Bearer $ApiKey"
            "Content-Type" = "multipart/form-data; boundary=$boundary"
        }
        $result = Invoke-RestMethod `
            -Uri "https://files.stripe.com/v1/files" `
            -Method POST `
            -Headers $uploadHeaders `
            -Body $body
        return $result.url
    } catch {
        $msg = $_.ErrorDetails.Message | ConvertFrom-Json -ErrorAction SilentlyContinue
        Write-Host "     Image upload failed: $($msg.error.message ?? $_)" -ForegroundColor Red
        return $null
    }
}

# ── Update product: description + metadata + image ────────────
function Update-Product {
    param(
        [string]$ProductId,
        [string]$Name,
        [string]$Description,
        [string]$StatementDescriptor,   # max 22 chars, no special chars
        [string]$UnitLabel,
        [string]$TaxCode,
        [hashtable]$Metadata,
        [string]$ImageFile              # filename only, e.g. "sub_creator_lite.png"
    )

    Write-Host "`n  Updating: $Name  ($ProductId)" -ForegroundColor Cyan

    # Upload image first
    $imageUrl = $null
    if ($ImageFile) {
        $fullPath = Join-Path $ImageFolder $ImageFile
        Write-Host "    Uploading image: $ImageFile" -ForegroundColor DarkGray
        $imageUrl = Upload-Image -FilePath $fullPath
        if ($imageUrl) {
            Write-Host "     Image uploaded: $imageUrl" -ForegroundColor DarkGreen
        }
    }

    # Build update body
    $body = @{
        name                  = $Name
        description           = $Description
        statement_descriptor  = $StatementDescriptor
        unit_label            = $UnitLabel
        "tax_code"            = $TaxCode
    }

    # Flatten metadata
    foreach ($kv in $Metadata.GetEnumerator()) {
        $body["metadata[$($kv.Key)]"] = $kv.Value
    }

    # Attach image if we got a URL
    if ($imageUrl) {
        $body["images[0]"] = $imageUrl
    }

    $result = Invoke-Stripe -Endpoint "products/$ProductId" -Body $body
    if ($result) {
        Write-Host "     Product updated" -ForegroundColor Green
    }
    return $result
}


# =============================================================
#  PRODUCT DEFINITIONS — IDs must match your Stripe Dashboard.
#  Each Price uses lookup_key from metadata (see stages/entitlements.py).
# =============================================================

$TAX = "txcd_10103001"   # SaaS / software as a service


# ─── SUBSCRIPTIONS ────────────────────────────────────────────

Update-Product `
    -ProductId           "prod_UDD0jSiHkf0s0n" `
    -Name                "UploadM8 Creator Lite — Monthly Subscription" `
    -Description         "UploadM8 Creator Lite is a recurring monthly SaaS subscription granting access to the UploadM8 multi-platform video publishing platform. Subscriber receives: 10 connected social accounts per platform (TikTok, YouTube Shorts, Instagram Reels, Facebook Reels), 500 PUT upload tokens per month included (released daily in UTC), 4,500 AIC AI-credit tokens per month included (released daily in UTC), 12-hour scheduling lookahead, queue depth of 100 jobs, no watermark on published content, webhook delivery, template library access, and enhanced AI content scanning. Unused included credits roll over in your wallet; top-up purchases never expire. Service is delivered digitally via uploadm8.com immediately upon payment confirmation. Subscription renews monthly. 7-day free trial included for new subscribers. Cancellation takes effect at end of current billing period — no partial refunds issued for unused days." `
    -StatementDescriptor "UPLOADM8 CREATOR LITE" `
    -UnitLabel           "subscription" `
    -TaxCode             $TAX `
    -Metadata            @{ type="subscription"; lookup_key="uploadm8_creator_lite_monthly"; tier="creator_lite"; put_monthly="500"; aic_monthly="4500"; max_accounts="10"; queue_depth="100"; lookahead_h="12" } `
    -ImageFile           "sub_creator_lite.png"


Update-Product `
    -ProductId           "prod_UDD01YK0Fa5SW5" `
    -Name                "UploadM8 Creator Pro — Monthly Subscription" `
    -Description         "UploadM8 Creator Pro is a recurring monthly SaaS subscription granting access to the UploadM8 multi-platform video publishing platform. Subscriber receives: 25 connected social accounts per platform (TikTok, YouTube Shorts, Instagram Reels, Facebook Reels), 1,800 PUT upload tokens per month included (released daily in UTC), 13,000 AIC AI-credit tokens per month included (released daily in UTC), 24-hour scheduling lookahead, queue depth of 500 jobs, no watermark on published content, priority processing lane, team seat access, webhook delivery, template library, and advanced AI content optimization. Unused included credits roll over in your wallet; top-up purchases never expire. Service is delivered digitally via uploadm8.com immediately upon payment confirmation. Subscription renews monthly. 7-day free trial included for new subscribers. Cancellation takes effect at end of current billing period — no partial refunds issued for unused days." `
    -StatementDescriptor "UPLOADM8 CREATOR PRO" `
    -UnitLabel           "subscription" `
    -TaxCode             $TAX `
    -Metadata            @{ type="subscription"; lookup_key="uploadm8_creator_pro_monthly"; tier="creator_pro"; put_monthly="1800"; aic_monthly="13000"; max_accounts="25"; queue_depth="500"; lookahead_h="24" } `
    -ImageFile           "sub_creator_pro.png"


Update-Product `
    -ProductId           "prod_UDD0NSQKtdQM7D" `
    -Name                "UploadM8 Studio — Monthly Subscription" `
    -Description         "UploadM8 Studio is a recurring monthly SaaS subscription granting access to the UploadM8 multi-platform video publishing platform. Subscriber receives: 75 connected social accounts per platform (TikTok, YouTube Shorts, Instagram Reels, Facebook Reels), 7,000 PUT upload tokens per month included (released daily in UTC), 45,000 AIC AI-credit tokens per month included (released daily in UTC), 72-hour scheduling lookahead, queue depth of 2,500 jobs, no watermark on published content, turbo throughput processing, analytics export, white-label publishing options, expanded team seats, webhook delivery, template library, and maximum-depth AI content optimization. Unused included credits roll over in your wallet; top-up purchases never expire. Service is delivered digitally via uploadm8.com immediately upon payment confirmation. Subscription renews monthly. 7-day free trial included for new subscribers. Cancellation takes effect at end of current billing period — no partial refunds issued for unused days." `
    -StatementDescriptor "UPLOADM8 STUDIO" `
    -UnitLabel           "subscription" `
    -TaxCode             $TAX `
    -Metadata            @{ type="subscription"; lookup_key="uploadm8_studio_monthly"; tier="studio"; put_monthly="7000"; aic_monthly="45000"; max_accounts="75"; queue_depth="2500"; lookahead_h="72" } `
    -ImageFile           "sub_studio.png"


Update-Product `
    -ProductId           "prod_UDD0JV6l2sbrmM" `
    -Name                "UploadM8 Agency — Monthly Subscription" `
    -Description         "UploadM8 Agency is a recurring monthly SaaS subscription granting access to the UploadM8 multi-platform video publishing platform at maximum scale. Subscriber receives: 300 connected social accounts per platform (TikTok, YouTube Shorts, Instagram Reels, Facebook Reels), 22,000 PUT upload tokens per month included (released daily in UTC), 140,000 AIC AI-credit tokens per month included (released daily in UTC), 168-hour (7-day) scheduling lookahead, unlimited queue depth, no watermark on published content, dedicated processing lane, white-label publishing, full team management, analytics export, webhook delivery, template library, and maximum-depth AI content optimization. Unused included credits roll over in your wallet; top-up purchases never expire. Service is delivered digitally via uploadm8.com immediately upon payment confirmation. Subscription renews monthly. 7-day free trial included for new subscribers. Cancellation takes effect at end of current billing period — no partial refunds issued for unused days." `
    -StatementDescriptor "UPLOADM8 AGENCY" `
    -UnitLabel           "subscription" `
    -TaxCode             $TAX `
    -Metadata            @{ type="subscription"; lookup_key="uploadm8_agency_monthly"; tier="agency"; put_monthly="22000"; aic_monthly="140000"; max_accounts="300"; queue_depth="unlimited"; lookahead_h="168" } `
    -ImageFile           "sub_agency.png"


# ─── PUT TOP-UPS (lookup keys must match stages/entitlements TOPUP_PRODUCTS) ──

Update-Product `
    -ProductId           "prod_UDD0VgzKZur5Qy" `
    -Name                "UploadM8 PUT 250 — Upload Token Top-Up" `
    -Description         "One-time purchase of 250 PUT (upload) tokens for the UploadM8 platform. Each PUT token authorises one multi-platform video publish event across any combination of connected accounts on TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are credited to the purchaser's UploadM8 wallet instantly upon payment confirmation and are available for use immediately. Tokens never expire and carry over month to month. This is a non-refundable digital goods purchase; once tokens are credited to the wallet they cannot be reversed. First-time purchaser bonus may apply. Service delivered digitally via uploadm8.com." `
    -StatementDescriptor "UPLOADM8 PUT 250" `
    -UnitLabel           "token" `
    -TaxCode             $TAX `
    -Metadata            @{ type="topup"; lookup_key="uploadm8_put_250"; wallet="put"; amount="250" } `
    -ImageFile           "topup_put_250.png"


Update-Product `
    -ProductId           "prod_UDCtVBGtibAVxy" `
    -Name                "UploadM8 PUT 500 — Upload Token Top-Up" `
    -Description         "One-time purchase of 500 PUT (upload) tokens for the UploadM8 platform. Each PUT token authorises one multi-platform video publish event across any combination of connected accounts on TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are credited to the purchaser's UploadM8 wallet instantly upon payment confirmation and are available for use immediately. Tokens never expire and carry over month to month. This is a non-refundable digital goods purchase; once tokens are credited to the wallet they cannot be reversed. First-time purchaser bonus may apply. Service delivered digitally via uploadm8.com." `
    -StatementDescriptor "UPLOADM8 PUT 500" `
    -UnitLabel           "token" `
    -TaxCode             $TAX `
    -Metadata            @{ type="topup"; lookup_key="uploadm8_put_500"; wallet="put"; amount="500" } `
    -ImageFile           "topup_put_500.png"


Update-Product `
    -ProductId           "prod_UDCtoSuZPbF0yl" `
    -Name                "UploadM8 PUT 1,000 — Upload Token Top-Up" `
    -Description         "One-time purchase of 1,000 PUT (upload) tokens for the UploadM8 platform. Each PUT token authorises one multi-platform video publish event across any combination of connected accounts on TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are credited to the purchaser's UploadM8 wallet instantly upon payment confirmation and are available for use immediately. Tokens never expire and carry over month to month. This is a non-refundable digital goods purchase; once tokens are credited to the wallet they cannot be reversed. First-time purchaser bonus may apply. Service delivered digitally via uploadm8.com." `
    -StatementDescriptor "UPLOADM8 PUT 1000" `
    -UnitLabel           "token" `
    -TaxCode             $TAX `
    -Metadata            @{ type="topup"; lookup_key="uploadm8_put_1000"; wallet="put"; amount="1000" } `
    -ImageFile           "topup_put_1000.png"


Update-Product `
    -ProductId           "prod_UDCtU5bvgTdZdn" `
    -Name                "UploadM8 PUT 2,500 — Upload Token Top-Up" `
    -Description         "One-time purchase of 2,500 PUT (upload) tokens for the UploadM8 platform. Each PUT token authorises one multi-platform video publish event across any combination of connected accounts on TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are credited to the purchaser's UploadM8 wallet instantly upon payment confirmation and are available for use immediately. Tokens never expire and carry over month to month. This is a non-refundable digital goods purchase; once tokens are credited to the wallet they cannot be reversed. First-time purchaser bonus may apply. Service delivered digitally via uploadm8.com." `
    -StatementDescriptor "UPLOADM8 PUT 2500" `
    -UnitLabel           "token" `
    -TaxCode             $TAX `
    -Metadata            @{ type="topup"; lookup_key="uploadm8_put_2500"; wallet="put"; amount="2500" } `
    -ImageFile           "topup_put_2500.png"


Update-Product `
    -ProductId           "prod_UDCtR9OeXCSlEB" `
    -Name                "UploadM8 PUT 5,000 — Upload Token Top-Up" `
    -Description         "One-time purchase of 5,000 PUT (upload) tokens for the UploadM8 platform. Each PUT token authorises one multi-platform video publish event across any combination of connected accounts on TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are credited to the purchaser's UploadM8 wallet instantly upon payment confirmation and are available for use immediately. Tokens never expire and carry over month to month. This is a non-refundable digital goods purchase; once tokens are credited to the wallet they cannot be reversed. First-time purchaser bonus may apply. Service delivered digitally via uploadm8.com." `
    -StatementDescriptor "UPLOADM8 PUT 5000" `
    -UnitLabel           "token" `
    -TaxCode             $TAX `
    -Metadata            @{ type="topup"; lookup_key="uploadm8_put_5000"; wallet="put"; amount="5000" } `
    -ImageFile           "topup_put_5000.png"


# ─── AIC TOP-UPS ──────────────────────────────────────────────

Update-Product `
    -ProductId           "prod_UDD0CCQzccfHTY" `
    -Name                "UploadM8 AIC 500 — AI Credit Token Top-Up" `
    -Description         "One-time purchase of 500 AIC (AI credit) tokens for the UploadM8 platform. AIC tokens power UploadM8's AI feature suite: automated caption generation, hashtag strategy optimisation, thumbnail image analysis, smart scheduling suggestions, and content metadata enrichment across TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are credited to the purchaser's UploadM8 wallet instantly upon payment confirmation and are available for use immediately. Tokens never expire and carry over month to month. This is a non-refundable digital goods purchase; once tokens are credited to the wallet they cannot be reversed. First-time purchaser bonus may apply. Service delivered digitally via uploadm8.com." `
    -StatementDescriptor "UPLOADM8 AIC 500" `
    -UnitLabel           "token" `
    -TaxCode             $TAX `
    -Metadata            @{ type="topup"; lookup_key="uploadm8_aic_500"; wallet="aic"; amount="500" } `
    -ImageFile           "topup_aic_500.png"


Update-Product `
    -ProductId           "prod_UDCt8PqZ9z1VnF" `
    -Name                "UploadM8 AIC 1,000 — AI Credit Token Top-Up" `
    -Description         "One-time purchase of 1,000 AIC (AI credit) tokens for the UploadM8 platform. AIC tokens power UploadM8's AI feature suite: automated caption generation, hashtag strategy optimisation, thumbnail image analysis, smart scheduling suggestions, and content metadata enrichment across TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are credited to the purchaser's UploadM8 wallet instantly upon payment confirmation and are available for use immediately. Tokens never expire and carry over month to month. This is a non-refundable digital goods purchase; once tokens are credited to the wallet they cannot be reversed. First-time purchaser bonus may apply. Service delivered digitally via uploadm8.com." `
    -StatementDescriptor "UPLOADM8 AIC 1000" `
    -UnitLabel           "token" `
    -TaxCode             $TAX `
    -Metadata            @{ type="topup"; lookup_key="uploadm8_aic_1000"; wallet="aic"; amount="1000" } `
    -ImageFile           "topup_aic_1000.png"


Update-Product `
    -ProductId           "prod_UDCt2xbikvyhjF" `
    -Name                "UploadM8 AIC 2,500 — AI Credit Token Top-Up" `
    -Description         "One-time purchase of 2,500 AIC (AI credit) tokens for the UploadM8 platform. AIC tokens power UploadM8's AI feature suite: automated caption generation, hashtag strategy optimisation, thumbnail image analysis, smart scheduling suggestions, and content metadata enrichment across TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are credited to the purchaser's UploadM8 wallet instantly upon payment confirmation and are available for use immediately. Tokens never expire and carry over month to month. This is a non-refundable digital goods purchase; once tokens are credited to the wallet they cannot be reversed. First-time purchaser bonus may apply. Service delivered digitally via uploadm8.com." `
    -StatementDescriptor "UPLOADM8 AIC 2500" `
    -UnitLabel           "token" `
    -TaxCode             $TAX `
    -Metadata            @{ type="topup"; lookup_key="uploadm8_aic_2500"; wallet="aic"; amount="2500" } `
    -ImageFile           "topup_aic_2500.png"


Update-Product `
    -ProductId           "prod_UDCt4lB4Dz0SCC" `
    -Name                "UploadM8 AIC 5,000 — AI Credit Token Top-Up" `
    -Description         "One-time purchase of 5,000 AIC (AI credit) tokens for the UploadM8 platform. AIC tokens power UploadM8's AI feature suite: automated caption generation, hashtag strategy optimisation, thumbnail image analysis, smart scheduling suggestions, and content metadata enrichment across TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are credited to the purchaser's UploadM8 wallet instantly upon payment confirmation and are available for use immediately. Tokens never expire and carry over month to month. This is a non-refundable digital goods purchase; once tokens are credited to the wallet they cannot be reversed. First-time purchaser bonus may apply. Service delivered digitally via uploadm8.com." `
    -StatementDescriptor "UPLOADM8 AIC 5000" `
    -UnitLabel           "token" `
    -TaxCode             $TAX `
    -Metadata            @{ type="topup"; lookup_key="uploadm8_aic_5000"; wallet="aic"; amount="5000" } `
    -ImageFile           "topup_aic_5000.png"


Update-Product `
    -ProductId           "prod_UDCtn73TukitZY" `
    -Name                "UploadM8 AIC 10,000 — AI Credit Token Top-Up" `
    -Description         "One-time purchase of 10,000 AIC (AI credit) tokens for the UploadM8 platform. AIC tokens power UploadM8's AI feature suite: automated caption generation, hashtag strategy optimisation, thumbnail image analysis, smart scheduling suggestions, and content metadata enrichment across TikTok, YouTube Shorts, Instagram Reels, and Facebook Reels. Tokens are credited to the purchaser's UploadM8 wallet instantly upon payment confirmation and are available for use immediately. Tokens never expire and carry over month to month. This is a non-refundable digital goods purchase; once tokens are credited to the wallet they cannot be reversed. First-time purchaser bonus may apply. Service delivered digitally via uploadm8.com." `
    -StatementDescriptor "UPLOADM8 AIC 10000" `
    -UnitLabel           "token" `
    -TaxCode             $TAX `
    -Metadata            @{ type="topup"; lookup_key="uploadm8_aic_10000"; wallet="aic"; amount="10000" } `
    -ImageFile           "topup_aic_10000.png"


Write-Host "`n=== All 14 Stripe products updated (4 subs + 10 top-ups) ===" -ForegroundColor Magenta
Write-Host "Create matching Prices in Stripe with the same lookup_key as metadata. Run with sk_live_... for production.`n" -ForegroundColor Yellow
