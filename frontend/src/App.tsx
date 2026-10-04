import { useEffect, useState } from 'react'
import Header, { type Route } from './components/Header'
import AnalyzePage from './pages/AnalyzePage'
import HistoryPage from './pages/HistoryPage'
import StatsPage from './pages/StatsPage'

const ROUTES: Route[] = ['analyze', 'history', 'stats']

function currentRoute(): Route {
  const r = window.location.hash.replace(/^#\/?/, '') as Route
  return ROUTES.includes(r) ? r : 'analyze'
}

export default function App() {
  const [route, setRoute] = useState<Route>(currentRoute)
  useEffect(() => {
    const on = () => setRoute(currentRoute())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])
  return (
    <div className="flex min-h-screen flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-10 focus:rounded focus:bg-[var(--surface-card)] focus:px-3 focus:py-2 focus:text-[var(--ink-primary)]"
      >
        Skip to content
      </a>
      <Header route={route} />
      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:py-8">
        {route === 'analyze' && <AnalyzePage />}
        {route === 'history' && <HistoryPage />}
        {route === 'stats' && <StatsPage />}
      </main>
      <footer
        className="border-t py-4 text-center text-xs text-[var(--ink-muted)]"
        style={{ borderColor: 'var(--line-hairline)', background: 'var(--surface-card)' }}
      >
        TriGuard · Verify before you apply · Report fraud:{' '}
        <a className="underline hover:text-[var(--ink-secondary)]" href="https://cybercrime.gov.in" target="_blank" rel="noreferrer">
          cybercrime.gov.in
        </a>{' '}
        / 1930
      </footer>
    </div>
  )
}
