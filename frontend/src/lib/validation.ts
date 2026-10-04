/** Client-side input validation (mirrors the backend limits; the server re-validates). */
export const MAX_IMAGE_BYTES = 5 * 1024 * 1024
export const MAX_TEXT_CHARS = 20_000
export const MAX_URL_CHARS = 2048
export const ALLOWED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp']

export function validateText(text: string): string | null {
  return text.length > MAX_TEXT_CHARS ? `Text is too long (max ${MAX_TEXT_CHARS.toLocaleString()} characters).` : null
}

export function validateUrl(raw: string): string | null {
  const value = raw.trim()
  if (!value) return null
  if (value.length > MAX_URL_CHARS) return 'This link is too long.'
  if (/\s/.test(value)) return 'A web address cannot contain spaces.'
  if (/^(javascript|data|mailto|file|ftp|tel):/i.test(value)) return 'Only http(s) web addresses are supported.'
  try {
    const url = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(value) ? value : `http://${value}`)
    if (!['http:', 'https:'].includes(url.protocol)) return 'Only http(s) web addresses are supported.'
    const host = url.hostname
    const isIp = /^\d{1,3}(\.\d{1,3}){3}$/.test(host) || host.includes(':')
    if (!host.includes('.') && !isIp) return 'Enter a full web address such as example.com/careers.'
    return null
  } catch {
    return 'This does not look like a valid web address.'
  }
}

export function validateImage(file: { type: string; size: number } | null | undefined): string | null {
  if (!file) return null
  if (!ALLOWED_IMAGE_TYPES.includes(file.type)) return 'Please choose a PNG, JPG or WebP image.'
  if (file.size > MAX_IMAGE_BYTES) return `Image is too large (max ${MAX_IMAGE_BYTES / (1024 * 1024)} MB).`
  if (file.size === 0) return 'This image file is empty.'
  return null
}

export interface FormState {
  text: string
  url: string
  image: File | null
}

export interface FormErrors {
  text: string | null
  url: string | null
  image: string | null
}

export function validateForm(s: FormState): FormErrors {
  return { text: validateText(s.text), url: validateUrl(s.url), image: validateImage(s.image) }
}

/** True when at least one input is provided. */
export function hasInput(s: FormState): boolean {
  return Boolean(s.text.trim() || s.url.trim() || s.image)
}

/** The analyze button is enabled only with an input and no validation errors. */
export function canSubmit(s: FormState): boolean {
  const e = validateForm(s)
  return hasInput(s) && !e.text && !e.url && !e.image
}
