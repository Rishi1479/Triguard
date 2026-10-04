import { useEffect, useState } from 'react'
import Header, { type Route } from './components/Header'
import AboutPage from './pages/AboutPage'
import AnalyzePage from './pages/AnalyzePage'
import HistoryPage from './pages/HistoryPage'

const ROUTES: Route[] = ['analyze', 'history', 'about']

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
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-10 focus:rounded focus:bg-white focus:px-3 focus:py-2">Skip to content</a>
      <Header route={route} />
      <main id="main" className="mx-auto w-full max-w-5xl flex-1 px-4 py-6">
        {route === 'analyze' && <AnalyzePage />}
        {route === 'history' && <HistoryPage />}
        {route === 'about' && <AboutPage />}
      </main>
      <footer className="border-t border-slate-200 bg-white py-4 text-center text-xs text-slate-500">
        TriGuard · Verify before you apply · Report fraud: cybercrime.gov.in / 1930
      </footer>
    </div>
  )
}
