import { ShieldIcon } from './Icons'

export type Route = 'analyze' | 'history' | 'about'
const LINKS: { route: Route; label: string }[] = [
  { route: 'analyze', label: 'Analyze' },
  { route: 'history', label: 'History' },
  { route: 'about', label: 'How it works' },
]

export default function Header({ route }: { route: Route }) {
  return (
    <header className="bg-navy-900 text-white">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-4 py-3">
        <a href="#/analyze" className="flex items-center gap-2 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-white">
          <ShieldIcon className="text-emerald-400" width={28} height={28} />
          <span>
            <span className="block text-xl font-bold leading-tight">TriGuard</span>
            <span className="block text-xs text-navy-100">Verify before you apply</span>
          </span>
        </a>
        <nav aria-label="Main">
          <ul className="flex gap-1">
            {LINKS.map((l) => (
              <li key={l.route}>
                <a
                  href={`#/${l.route}`}
                  aria-current={route === l.route ? 'page' : undefined}
                  className={`block rounded-md px-3 py-1.5 text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-white ${
                    route === l.route ? 'bg-navy-700 text-white' : 'text-navy-100 hover:bg-navy-800'
                  }`}
                >
                  {l.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      </div>
    </header>
  )
}
