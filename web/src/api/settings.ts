// The caller's own settings: GET/PUT /dash/api/settings/image (mcp/server.py,
// mcp/images.py settings/set_setting). The key decides whose account is read
// and written; nothing here names an account.

export const IMAGE_SETTINGS_PATH = '/dash/api/settings/image';

export type ImageModelOption = {
  id: string;
  name: string;
  label: string;
  steps: number;
  est_seconds: number | null;
  est_basis: string;
  licence: string;
};

export type ImageSettings = {
  ok: true;
  /** null: this account never chose, so the server default applies */
  choice: string | null;
  default: string;
  /** what this account's next image uses */
  effective: string;
  options: ImageModelOption[];
};

/** "~24 s · n=1 smoke test, sd-cli", or the basis alone when unmeasured. */
export function secondsText(o: ImageModelOption): string {
  if (o.est_seconds === null || o.est_seconds === undefined) return o.est_basis || 'not measured';
  return `~${Math.round(o.est_seconds)} s · ${o.est_basis}`;
}

/** The body a pick sends. `null` returns the account to the server default. */
export function choiceBody(id: string | null): { choice: string | null } {
  return { choice: id };
}
