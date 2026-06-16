import sys

file_path = '/home/alli-ekundayo/Projects/Ego/frontend/src/App.jsx'
with open(file_path, 'r') as f:
    content = f.read()

# 1. Spread elements
content = content.replace('max-w-7xl', 'max-w-[85rem]')
content = content.replace('72rem', '85rem')

# 2. Highlight text on their respective pages
content = content.replace(
    'Task A dashboard for user modelling.',
    '<span className="text-emerald-600 dark:text-emerald-450 italic font-serif">Task A</span> dashboard for user modelling.'
)

content = content.replace(
    'Task B dashboard for contextual recommendations.',
    '<span className="text-orange-600 dark:text-orange-450 italic font-serif">Task B</span> dashboard for contextual recommendations.'
)

# 3. Update UserDashboard to accept color
old_user_dashboard_start = """function UserDashboard({ title, users, loading, selectedUserId, onChange, selectedUser }) {"""

new_user_dashboard_start = """function UserDashboard({ title, users, loading, selectedUserId, onChange, selectedUser, color = "emerald" }) {
  const colorMap = {
    emerald: {
      badgeBorder: "border-emerald-200/70 dark:border-emerald-900/50",
      badgeBg: "bg-emerald-50/80 dark:bg-emerald-950/40",
      badgeText: "text-emerald-700 dark:text-emerald-400",
      statText: "text-emerald-600 dark:text-emerald-450",
    },
    orange: {
      badgeBorder: "border-orange-200/70 dark:border-orange-900/50",
      badgeBg: "bg-orange-50/80 dark:bg-orange-950/40",
      badgeText: "text-orange-700 dark:text-orange-400",
      statText: "text-orange-600 dark:text-orange-450",
    }
  };
  const themeStyles = colorMap[color] || colorMap.emerald;"""

content = content.replace(old_user_dashboard_start, new_user_dashboard_start)

# Replace the specific hardcoded emerald classes in UserDashboard
old_ud_badge = """<div className="inline-flex items-center gap-2 rounded-full border border-emerald-200/70 dark:border-emerald-900/50 bg-emerald-50/80 dark:bg-emerald-950/40 px-3 py-1 text-[10px] uppercase tracking-[0.22em] text-emerald-700 dark:text-emerald-400">"""
new_ud_badge = """<div className={`inline-flex items-center gap-2 rounded-full border ${themeStyles.badgeBorder} ${themeStyles.badgeBg} px-3 py-1 text-[10px] uppercase tracking-[0.22em] ${themeStyles.badgeText}`}>"""
content = content.replace(old_ud_badge, new_ud_badge)

old_ud_stat = """<p className="mt-2 text-2xl font-semibold text-emerald-600 dark:text-emerald-450">"""
new_ud_stat = """<p className={`mt-2 text-2xl font-semibold ${themeStyles.statText}`}>"""
content = content.replace(old_ud_stat, new_ud_stat)

# Update the TaskBPage call to UserDashboard
old_tb_call = """<UserDashboard
        title="Select a user profile"
        users={users}
        loading={loading}
        selectedUserId={effectiveSelectedUserId}
        onChange={setSelectedUserId}
        selectedUser={selectedUser}
      />"""
new_tb_call = """<UserDashboard
        title="Select a user profile"
        users={users}
        loading={loading}
        selectedUserId={effectiveSelectedUserId}
        onChange={setSelectedUserId}
        selectedUser={selectedUser}
        color="orange"
      />"""
content = content.replace(old_tb_call, new_tb_call)

# Fix 'See Task A' button in TaskBPage which still uses emerald
old_see_task_a = """<AppLink
            className="inline-flex items-center gap-2 rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-5 py-3 text-sm font-semibold text-zinc-700 dark:text-zinc-300 transition-all hover:-translate-y-0.5 hover:border-zinc-300/70 dark:hover:border-zinc-700 hover:text-zinc-950 dark:hover:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500/40 focus:ring-offset-2"
            href="/task-a"
          >
            See Task A
            <ArrowRight size={14} weight="bold" className="text-emerald-600" />
          </AppLink>"""
new_see_task_a = """<AppLink
            className="inline-flex items-center gap-2 rounded-xl border border-zinc-200/70 dark:border-zinc-800 bg-white dark:bg-zinc-900 px-5 py-3 text-sm font-semibold text-zinc-700 dark:text-zinc-300 transition-all hover:-translate-y-0.5 hover:border-zinc-300/70 dark:hover:border-zinc-700 hover:text-zinc-950 dark:hover:text-white focus:outline-none focus:ring-2 focus:ring-orange-500/40 focus:ring-offset-2"
            href="/task-a"
          >
            See Task A
            <ArrowRight size={14} weight="bold" className="text-orange-600" />
          </AppLink>"""
content = content.replace(old_see_task_a, new_see_task_a)

with open(file_path, 'w') as f:
    f.write(content)

print("Done")
