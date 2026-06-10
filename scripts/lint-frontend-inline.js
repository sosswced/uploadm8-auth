#!/usr/bin/env node
/**
 * Lint inline <script> blocks in frontend HTML for syntax errors and risky patterns.
 * Usage: node scripts/lint-frontend-inline.js
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..', 'frontend');
const PAGES = [
    'queue.html',
    'dashboard.html',
    'upload.html',
    'analytics.html',
    'kpi.html',
    'scheduled.html',
    'admin.html',
    'admin-kpi.html',
    'admin-marketing.html',
    'admin-ml-observability.html',
    'settings.html',
    'smart-insights.html',
    'platforms.html',
    'groups.html',
    'thumbnail-studio.html',
];

const RISKY = [
    {
        name: 'object_literal_property_access',
        re: /,\s*[a-zA-Z_$][\w$]*\s*,\s*[a-zA-Z_$][\w$]*\.[a-zA-Z_$][\w$]*\s*[|,]/,
        hint: 'Invalid object literal — use caption: payload.caption not ,title,payload.caption',
    },
];

let errors = 0;
let warnings = 0;

for (const page of PAGES) {
    const file = path.join(ROOT, page);
    if (!fs.existsSync(file)) {
        console.warn('skip missing', page);
        continue;
    }
    const html = fs.readFileSync(file, 'utf8');
    const scripts = [...html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi)];
    scripts.forEach((m, i) => {
        const js = m[1].trim();
        if (!js) return;
        try {
            // eslint-disable-next-line no-new-func
            new Function(js);
        } catch (e) {
            console.error(`${page}: inline script #${i}: SYNTAX ${e.message}`);
            errors += 1;
        }
        RISKY.forEach((rule) => {
            if (rule.re.test(js)) {
                console.warn(`${page}: inline script #${i}: WARN ${rule.name} — ${rule.hint}`);
                warnings += 1;
            }
        });
    });
}

if (errors) {
    console.error(`frontend inline lint: ${errors} error(s), ${warnings} warning(s)`);
    process.exit(1);
}
console.log(`frontend inline lint: OK (${PAGES.length} pages, ${warnings} warning(s))`);
process.exit(0);
