(function () {
  let selectedFormatKey = null;
  let lastJobId = null;
  let cachedVariants = [];
  let personaImageDataUrls = [];

  function el(id) { return document.getElementById(id); }
  function esc(s) { return String(s || "").replace(/[&<>"']/g, (c) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[c])); }

  /** Plain-language line for the “engine mode” slug returned by the API. */
  function humanEngineMode(mode) {
    const m = String(mode || "");
    if (m === "uploadm8_pikzels_v2_r2") return "Full AI previews (Pikzels)";
    if (m === "uploadm8_heuristic_youtube_ref_only") return "Layout ideas (YouTube reference only)";
    if (m === "uploadm8_heuristic") return "Layout ideas (built-in heuristics)";
    return "";
  }

  /** Readable layout hints instead of raw field names. */
  function layoutTipsLine(v) {
    const raw = Number(v.face_scale || 0);
    const fpct = Number.isFinite(raw) && raw <= 1 ? Math.round(raw * 100) : Math.round(raw);
    const pos = String(v.text_position || "").replace(/_/g, " ").trim();
    const c = String(v.contrast_profile || "").replace(/_/g, " ").trim().toLowerCase();
    const contrastPretty = (
      { "very high": "very strong contrast", high: "strong contrast", medium: "balanced contrast", low: "softer contrast" }[c]
    ) || (c ? `${c} contrast` : "");
    const parts = [];
    if (fpct > 0) parts.push(`face fills about ${fpct}% of the frame`);
    if (pos) parts.push(`text toward the ${pos}`);
    if (contrastPretty) parts.push(contrastPretty);
    return parts.join(" · ");
  }

  /** What went wrong with the preview (never show raw slugs like pikzels_recreate_error). */
  function previewStatusMessage(v) {
    const s = String(v.engine_status || "");
    if (s === "ok") return v.preview_url ? "" : "Preview is ready above.";
    if (s === "pikzels_recreate_error") {
      const detail = String(v.engine_error || "").trim();
      const base = "We couldn't generate the AI preview image. Check billing credits and your Pikzels API key on the server, then try again.";
      if (detail && detail.length < 420) {
        return `${base} (${detail})`;
      }
      if (detail) {
        return `${base} (${detail.slice(0, 400)}…)`;
      }
      return base;
    }
    if (s === "pikzels_no_image_bytes") return "The preview service returned an empty image — try Generate again.";
    if (s === "r2_upload_failed") return "We couldn't save the preview — wait a moment and try again.";
    if (s === "skipped_no_video_id") return "Use a valid YouTube link so we can use its cover as a reference.";
    if (s) return "Something went wrong while building this preview — try again or contact support with your job id.";
    return "";
  }

  async function toDataUrls(files) {
    const arr = Array.from(files || []).slice(0, 20);
    const out = [];
    for (const f of arr) {
      if (!f || !String(f.type || "").startsWith("image/")) continue;
      const data = await new Promise((resolve, reject) => {
        const fr = new FileReader();
        fr.onload = () => resolve(fr.result);
        fr.onerror = reject;
        fr.readAsDataURL(f);
      });
      out.push(String(data || ""));
    }
    return out;
  }

  async function loadFormats() {
    let rows = [];
    try {
      const niche = el("nicheInput").value;
      const resp = await apiCall(
        `/api/entitlements/thumbnail-studio-formats?niche=${encodeURIComponent(niche)}`
      );
      rows = (resp && resp.formats) || [];
    } catch (err) {
      const msg = err && err.message ? err.message : "Could not load layout formats.";
      if (typeof showToast === "function") showToast(msg, "warning");
      rows = [];
    }
    const keys = new Set(rows.map((r) => String(r.key || "")));
    if (selectedFormatKey && !keys.has(selectedFormatKey)) selectedFormatKey = null;
    if (!selectedFormatKey && rows.length) selectedFormatKey = String(rows[0].key || "") || null;
    const host = el("formatChips");
    host.innerHTML = rows.map((r) => (
      `<button class="format-chip ${selectedFormatKey === r.key ? "active" : ""}" data-key="${esc(r.key)}" type="button">
        <strong>${esc(r.name)}</strong> <span class="muted">${esc(r.social_proof || "")}</span>
      </button>`
    )).join("");
    host.querySelectorAll(".format-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        selectedFormatKey = btn.getAttribute("data-key");
        host.querySelectorAll(".format-chip").forEach((x) => x.classList.remove("active"));
        btn.classList.add("active");
      });
    });
  }

  async function loadPersonas() {
    let rows = [];
    try {
      const resp = await apiCall("/api/thumbnail-studio/personas");
      rows = (resp && resp.personas) || [];
    } catch (err) {
      const msg = err && err.message ? err.message : "Could not load personas.";
      if (typeof showToast === "function") showToast(msg, "warning");
      rows = [];
    }
    const sel = el("personaSelect");
    sel.innerHTML = `<option value="">No persona</option>` + rows.map((p) => {
      const n = Number(p.image_count || 0);
      const pk =
        p.pikzels_linked === true ? " · Pikzels linked" :
        p.pikzels_linked === false ? " · Pikzels not linked" : "";
      return `<option value="${esc(p.id)}">${esc(p.name)} (${n} photos${pk})</option>`;
    }).join("");
  }

  function getPayload() {
    return {
      youtube_url: el("youtubeUrl").value.trim(),
      topic: el("topicInput").value.trim(),
      niche: el("nicheInput").value,
      closeness: Number(el("closeness").value || 55),
      variant_count: Number(el("variantCount").value || 6),
      persona_id: el("personaSelect").value || null,
      format_key: selectedFormatKey,
      competitor_gap_mode: !!el("gapMode").checked,
    };
  }

  async function estimate() {
    try {
      const p = getPayload();
      const resp = await apiCall("/api/thumbnail-studio/estimate", {
        method: "POST",
        body: JSON.stringify({
          variant_count: p.variant_count,
          has_persona: !!p.persona_id,
          competitor_gap_mode: p.competitor_gap_mode,
          has_channel_memory: true,
        }),
      });
      el("costEstimateText").textContent = `Estimated: ${resp.put_cost} PUT + ${resp.aic_cost} AIC`;
    } catch (err) {
      const msg = err && err.message ? err.message : "Could not estimate cost.";
      if (typeof showToast === "function") showToast(msg, "warning");
    }
  }

  const _PIXEL_GIF =
    "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7";

  async function hydratePikzelsCdnPreviews(root) {
    if (!root || typeof window.apiFetch !== "function") return;
    const imgs = root.querySelectorAll("img.variant-thumb[data-variant-id]");
    for (const img of imgs) {
      const vid = img.getAttribute("data-variant-id");
      if (!vid) continue;
      try {
        const path =
          "/api/thumbnail-studio/cdn-preview?variant_id=" + encodeURIComponent(vid);
        const r = await window.apiFetch(path, {
          method: "GET",
          authRedirectOn401: true,
          timeoutMs: 120000,
        });
        if (!r || !r.ok) continue;
        const blob = await r.blob();
        if (blob && blob.size > 512) {
          img.src = URL.createObjectURL(blob);
        }
      } catch (e) {
        console.warn("[thumbnail-studio] cdn preview hydrate", e);
      }
    }
  }

  function renderVariants(variants) {
    cachedVariants = Array.isArray(variants) ? variants : [];
    const root = el("variants");
    root.innerHTML = cachedVariants.map((v) => {
      const score = Number(v.ctr_score || 0).toFixed(1);
      const layoutLine = layoutTipsLine(v);
      const previewNote = previewStatusMessage(v);
      const scoreTitle = "Model estimate from our thumbnail scorer — not your real YouTube CTR until you publish and measure.";
      const pu = String(v.preview_url || v.pikzels_cdn_url || "").trim();
      const isCdn = pu.indexOf("cdn.pikzels.com") >= 0;
      const vid = String(v.variant_id || "").trim();
      const imgBlock = pu
        ? `<div class="mt-2"><img class="variant-thumb" width="1280" height="720" style="width:100%;max-height:200px;object-fit:cover;border-radius:var(--radius-sm,6px);border:1px solid var(--border-color);" alt="" loading="lazy" ${isCdn && vid ? `data-variant-id="${esc(vid)}" src="${_PIXEL_GIF}"` : `src="${esc(pu)}"`} /></div>`
        : "";
      return (
      `<article class="variant-card">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:.5rem;">
          <strong>${esc(v.name || `Variant ${v.index || ""}`)}</strong>
          <span class="score-pill" title="${esc(scoreTitle)}">Score ${esc(score)}</span>
        </div>
        ${imgBlock}
        <p class="mt-2"><strong>${esc(v.headline || "")}</strong></p>
        <p class="muted">${esc(v.subhead || "")}</p>
        ${layoutLine ? `<p class="muted mt-2">${esc(layoutLine)}</p>` : ""}
        ${previewNote ? `<p class="muted mt-2" style="border-left:3px solid var(--accent-orange, #f97316);padding-left:.5rem;">${esc(previewNote)}</p>` : ""}
        <p class="muted mt-1" style="font-size:.78rem;">Tips</p>
        <ul class="muted mt-1">${(v.suggestions || []).map((s) => `<li>${esc(s)}</li>`).join("")}</ul>
        <button class="btn btn-secondary mt-2 pick-variant" data-index="${Number(v.index || 1)}" type="button">Select Winner</button>
      </article>`
      );
    }).join("");
    hydratePikzelsCdnPreviews(root).catch(function () {});
    root.querySelectorAll(".pick-variant").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!lastJobId) return;
        const idx = Number(btn.getAttribute("data-index") || 1);
        const hit = cachedVariants.find((x) => Number(x.index) === idx);
        if (!hit || !hit.variant_id) return;
        try {
          await apiCall("/api/thumbnail-studio/feedback", {
            method: "POST",
            body: JSON.stringify({
              job_id: lastJobId,
              variant_id: hit.variant_id,
              event_type: "selected",
              metadata: { source: "thumbnail_studio_ui" },
            }),
          });
          showToast("Got it — we’ll lean on this pick for future suggestions on your channel.", "success");
        } catch (err) {
          const msg = err && err.message ? err.message : "Could not record your pick.";
          if (typeof showToast === "function") showToast(msg, "warning");
        }
      });
    });
  }

  async function generate() {
    const p = getPayload();
    if (!p.youtube_url) {
      showToast("Paste a YouTube URL first.", "warning");
      return;
    }
    try {
      el("generateBtn").disabled = true;
      const res = await apiCall("/api/thumbnail-studio/recreate", {
        method: "POST",
        body: JSON.stringify(p),
      });
      lastJobId = res.job_id;
      renderVariants((res.variants || []).map((v) => ({ ...v, variant_id: v.variant_id || null })));
      el("abExportBtn").disabled = false;
      const eng = humanEngineMode(res.engine_mode);
      const ai = res.m8_engine && res.m8_engine.ai_display ? String(res.m8_engine.ai_display) : "";
      const parts = [`Charged: ${res.put_cost} PUT + ${res.aic_cost} AIC`];
      if (eng) parts.push(eng);
      if (ai) parts.push(ai);
      el("costEstimateText").textContent = parts.join(" · ");

      const job = await apiCall(`/api/thumbnail-studio/jobs/${encodeURIComponent(lastJobId)}`);
      const jobVariants = (job && job.variants) || [];
      if (jobVariants.length) renderVariants(jobVariants);
      await refreshSavedJobsList();
    } catch (err) {
      const msg = err && err.message ? err.message : "Generation failed.";
      if (typeof showToast === "function") showToast(msg, "warning");
    } finally {
      el("generateBtn").disabled = false;
    }
  }

  async function linkSelectedPersonaPikzels() {
    const id = (el("personaSelect").value || "").trim();
    if (!id) {
      showToast("Choose a saved persona in the list first.", "warning");
      return;
    }
    const btn = el("linkPersonaPikzelsBtn");
    const prev = btn ? btn.textContent : "";
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Linking…";
    }
    try {
      if (typeof showToast === "function") {
        showToast("Sending photos to Pikzels — this may take a few minutes. Please keep this tab open.", "info", 12000);
      }
      const resp = await apiCall(
        `/api/thumbnail-studio/personas/${encodeURIComponent(id)}/link-pikzels`,
        { method: "POST" }
      );
      if (resp && resp.already_linked && typeof showToast === "function") {
        showToast("This persona is already linked to Pikzels.", "success");
      } else if (resp && resp.pikzels_linked && typeof showToast === "function") {
        showToast("Linked to Pikzels — you can use this persona for AI thumbnails.", "success");
      } else if (resp && resp.pikzels_warning && typeof showToast === "function") {
        showToast("Pikzels: " + String(resp.pikzels_warning), "warning", 14000);
      } else if (typeof showToast === "function") {
        showToast("Could not complete Pikzels link.", "warning");
      }
      await loadPersonas();
    } catch (err) {
      const msg = err && err.message ? err.message : "Link to Pikzels failed.";
      if (typeof showToast === "function") showToast(msg, "warning");
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = prev || "Link selected persona to Pikzels";
      }
    }
  }

  async function savePersona() {
    const name = el("personaName").value.trim();
    if (!name) {
      showToast("Persona name is required.", "warning");
      return;
    }
    if (personaImageDataUrls.length < 3 || personaImageDataUrls.length > 20) {
      showToast("Select 3-20 selfie images.", "warning");
      return;
    }
    try {
      const resp = await apiCall("/api/thumbnail-studio/personas", {
        method: "POST",
        body: JSON.stringify({
          name,
          image_urls: personaImageDataUrls,
        }),
      });
      if (resp && resp.pikzels_warning && typeof showToast === "function") {
        showToast("Saved; Pikzels: " + String(resp.pikzels_warning), "warning");
      } else if (resp && resp.pikzels_pikzonality_linked && typeof showToast === "function") {
        showToast("Persona saved and linked to Pikzels for recreate & uploads.", "success");
      } else if (typeof showToast === "function") {
        showToast("Persona saved.", "success");
      }
      await loadPersonas();
    } catch (err) {
      const msg = err && err.message ? err.message : "Could not save persona.";
      if (typeof showToast === "function") showToast(msg, "warning");
    }
  }

  function formatSavedJobLabel(j) {
    if (!j || !j.job_id) return "";
    const t = (j.created_at || "").replace("T", " ").slice(0, 16);
    const title = (j.source_title || j.topic || "Untitled").slice(0, 48);
    const niche = j.niche || "";
    return `${t} · ${title}${niche ? " · " + niche : ""}`;
  }

  async function refreshSavedJobsList() {
    const sel = el("savedJobSelect");
    if (!sel) return;
    try {
      const resp = await apiCall("/api/thumbnail-studio/jobs?limit=40");
      const rows = (resp && resp.jobs) || [];
      const cur = sel.value;
      sel.innerHTML = '<option value="">— Pick a saved job —</option>' + rows.map((j) => (
        `<option value="${esc(j.job_id)}">${esc(formatSavedJobLabel(j))}</option>`
      )).join("");
      if (cur && rows.some((j) => String(j.job_id) === cur)) sel.value = cur;
    } catch (err) {
      console.warn("[thumbnail-studio] saved jobs list", err);
    }
  }

  async function loadSelectedSavedJob() {
    const sel = el("savedJobSelect");
    if (!sel || !sel.value) {
      if (typeof showToast === "function") showToast("Choose a job from the list first.", "warning");
      return;
    }
    const jid = sel.value.trim();
    try {
      const job = await apiCall(`/api/thumbnail-studio/jobs/${encodeURIComponent(jid)}`);
      const meta = (job && job.job) || {};
      lastJobId = job.job_id || jid;
      el("youtubeUrl").value = String(meta.youtube_url || "");
      el("topicInput").value = String(meta.topic || "");
      el("nicheInput").value = String(meta.niche || "general");
      el("closeness").value = String(Number(meta.closeness) || 55);
      el("closenessLabel").textContent = String(el("closeness").value);
      const vc = Math.max(4, Math.min(8, parseInt(meta.variant_count, 10) || 6));
      el("variantCount").value = String(vc);
      el("gapMode").checked = !!meta.competitor_gap_mode;
      const pid = meta.persona_id ? String(meta.persona_id) : "";
      if (pid) {
        const ps = el("personaSelect");
        const opt = Array.from(ps.options || []).find((o) => o.value === pid);
        if (opt) ps.value = pid;
      }
      await loadFormats();
      const jobVariants = (job && job.variants) || [];
      renderVariants(jobVariants.map((v) => ({ ...v, variant_id: v.variant_id || null })));
      el("abExportBtn").disabled = !lastJobId;
      const put = Number(meta.put_cost) || 0;
      const aic = Number(meta.aic_cost) || 0;
      el("costEstimateText").textContent =
        `Loaded saved analysis (${put} PUT + ${aic} AIC were charged when this job originally ran).`;
      if (typeof showToast === "function") showToast("Loaded your saved analysis.", "success");
    } catch (err) {
      const msg = err && err.message ? err.message : "Could not load that job.";
      if (typeof showToast === "function") showToast(msg, "warning");
    }
  }

  async function exportAB() {
    if (!lastJobId) return;
    try {
      const res = await apiCall(`/api/thumbnail-studio/ab-export/${encodeURIComponent(lastJobId)}`);
      const dl = res.download_url || ((res.exports || [])[0] && res.exports[0].download_url);
      if (dl) {
        const a = document.createElement("a");
        a.href = dl;
        a.download = res.filename || "thumbnail_ab.zip";
        a.rel = "noopener";
        a.target = "_blank";
        document.body.appendChild(a);
        a.click();
        a.remove();
      }
      const msg = (res.exports || []).map((x) => `${x.label}: ${x.filename}`).join(" | ");
      const note = res.note ? ` ${res.note}` : "";
      showToast(dl ? `Downloading ${msg}.${note}` : `Comparison pack: ${msg}.${note}`, "info", 9000);
    } catch (err) {
      const msg = err && err.message ? err.message : "Export not available.";
      if (typeof showToast === "function") showToast(msg, "warning");
    }
  }

  async function init() {
    const user = await initApp("thumbnail-studio");
    if (!user) return;

    await Promise.all([loadFormats(), loadPersonas(), refreshSavedJobsList()]);
    el("nicheInput").addEventListener("change", loadFormats);
    el("estimateBtn").addEventListener("click", estimate);
    el("generateBtn").addEventListener("click", generate);
    el("abExportBtn").addEventListener("click", exportAB);
    if (el("loadSavedJobBtn")) el("loadSavedJobBtn").addEventListener("click", () => { loadSelectedSavedJob().catch(function () {}); });
    if (el("refreshSavedJobsBtn")) el("refreshSavedJobsBtn").addEventListener("click", () => { refreshSavedJobsList().catch(function () {}); });
    el("savePersonaBtn").addEventListener("click", savePersona);
    if (el("linkPersonaPikzelsBtn")) el("linkPersonaPikzelsBtn").addEventListener("click", linkSelectedPersonaPikzels);

    el("closeness").addEventListener("input", () => {
      el("closenessLabel").textContent = String(el("closeness").value);
    });
    el("personaFiles").addEventListener("change", async (e) => {
      personaImageDataUrls = await toDataUrls(e.target.files || []);
      showToast(`${personaImageDataUrls.length} photo(s) ready to save with this persona.`, "info");
    });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
