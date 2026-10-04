import { describe, expect, it } from 'vitest'
import { APP_TITLE } from './title'

describe('app', () => {
  it('has the expected title', () => {
    expect(APP_TITLE).toBe('TriGuard')
  })
})
