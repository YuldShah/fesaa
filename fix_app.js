const fs = require('fs');
const file = 'C:/Users/Based/Desktop/Project/bot/mini_app/frontend/src/App.jsx';
let content = fs.readFileSync(file, 'utf8');

const regex = /\/\/ â”€â”€â”€ Moderator app shell[\s\S]*?(?=\/\/ â”€â”€â”€ Regular user app shell)/;
const replacement = `// ─── Moderator app shell ───────────────────────────────────────────────────────────
function ModeratorApp() {
  const location = useLocation()
  const navigate = useNavigate()
  const { title, subtitle } = useModHeaderInfo()

  return (
    <div className="page-wrapper">
      <AppHeader title={title} subtitle={subtitle} />
      <nav className="admin-tab-nav">
        {[{ path: '/admin/topics', label: 'Topics', Icon: BookOpen }].map(({ path, label, Icon }) => {
          const active = location.pathname.startsWith(path)
          return (
            <button
              key={path}
              onClick={() => navigate(path)}
              className={\`admin-tab-item\${active ? ' active' : ''}\`}
            >
              <Icon size={18} strokeWidth={active ? 2.2 : 1.6} />
              {label}
            </button>
          )
        })}
      </nav>

      <div className="admin-scroll-container">
        <AnimatePresence mode="wait">
          <Suspense fallback={<PageLoader />}>
            <Routes location={location} key={location.pathname}>
              <Route path="/admin/topics"     element={<AnimatedPage><AdminTopics /></AnimatedPage>} />
              <Route path="/admin/topics/:id" element={<AnimatedPage><AdminTopicEdit /></AnimatedPage>} />
              <Route path="*"                 element={<Navigate to="/admin/topics" replace />} />
            </Routes>
          </Suspense>
        </AnimatePresence>
      </div>
    </div>
  )
}

`;
content = content.replace(regex, replacement);
fs.writeFileSync(file, content, 'utf8');
