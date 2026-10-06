const USER_KEY = "iota_user_id";

export async function getOrCreateUserId(): Promise<string> {
  if (typeof window === "undefined") return "server";

  const cached = localStorage.getItem(USER_KEY);
  if (cached) return cached;

  try {
    const FingerprintJS = await import("@fingerprintjs/fingerprintjs");
    const fp = await FingerprintJS.load();
    const result = await fp.get();
    const id = result.visitorId;
    localStorage.setItem(USER_KEY, id);
    return id;
  } catch {
    const fallback = crypto.randomUUID();
    localStorage.setItem(USER_KEY, fallback);
    return fallback;
  }
}

export function resetIdentity() {
  localStorage.removeItem(USER_KEY);
}
