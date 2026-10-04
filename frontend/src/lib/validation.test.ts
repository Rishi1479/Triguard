import { describe, expect, it } from 'vitest'
import { canSubmit, hasInput, MAX_IMAGE_BYTES, MAX_TEXT_CHARS, validateImage, validateText, validateUrl } from './validation'

const img = (type: string, size: number) => ({ type, size })
const file = (type = 'image/png', size = 1000) => new File([new Uint8Array(size)], 'a.png', { type })

describe('validateUrl', () => {
  it.each(['', '   ', 'https://www.tcs.com/careers', 'tcs.com/careers', 'http://192.168.1.5/job', 'forms.gle/AbC123'])('accepts %j', (u) => {
    expect(validateUrl(u)).toBeNull()
  })
  it.each(['not a url', 'javascript:alert(1)', 'mailto:a@b.com', 'ftp://x.com/a', 'localhost', 'http://', 'hello'])('rejects %j', (u) => {
    expect(validateUrl(u)).toBeTruthy()
  })
  it('rejects very long urls', () => expect(validateUrl('http://a.com/' + 'x'.repeat(3000))).toMatch(/too long/))
})

describe('validateImage', () => {
  it('accepts png/jpeg/webp within limit', () => {
    for (const t of ['image/png', 'image/jpeg', 'image/webp']) expect(validateImage(img(t, 1000))).toBeNull()
    expect(validateImage(null)).toBeNull()
  })
  it('rejects wrong type, oversize and empty', () => {
    expect(validateImage(img('application/pdf', 1000))).toMatch(/PNG, JPG or WebP/)
    expect(validateImage(img('image/gif', 1000))).toBeTruthy()
    expect(validateImage(img('image/png', MAX_IMAGE_BYTES + 1))).toMatch(/too large/)
    expect(validateImage(img('image/png', 0))).toMatch(/empty/)
  })
})

describe('validateText / submit gating', () => {
  it('limits text length', () => {
    expect(validateText('a'.repeat(MAX_TEXT_CHARS))).toBeNull()
    expect(validateText('a'.repeat(MAX_TEXT_CHARS + 1))).toMatch(/too long/)
  })
  it('requires at least one input', () => {
    expect(hasInput({ text: '  ', url: '', image: null })).toBe(false)
    expect(canSubmit({ text: '  ', url: '', image: null })).toBe(false)
  })
  it('any single input enables submit; any error disables it', () => {
    expect(canSubmit({ text: 'hello', url: '', image: null })).toBe(true)
    expect(canSubmit({ text: '', url: 'tcs.com', image: null })).toBe(true)
    expect(canSubmit({ text: '', url: '', image: file() })).toBe(true)
    expect(canSubmit({ text: 'hello', url: 'not a url', image: null })).toBe(false)
    expect(canSubmit({ text: 'hello', url: '', image: file('text/plain') })).toBe(false)
  })
})
