from dotenv import load_dotenv; load_dotenv()
import os, requests
h = {"Authorization": f"token {os.environ['GITHUB_APP_TOKEN']}", "Accept": "application/vnd.github+json"}
b = "https://api.github.com/repos/Grey30-2003dc/Nexgen_Demo"
prs = requests.get(f"{b}/pulls?state=all&sort=created&direction=desc", headers=h).json()
for pr in prs[:3]:
    n = pr["number"]
    print(f"\nPR #{n}: {pr['title']!r} | {pr['state']} | created {pr['created_at']} | head {pr['head']['sha'][:8]}")
    reviews = requests.get(f"{b}/pulls/{n}/reviews", headers=h).json()
    print("  reviews        :", [(r["submitted_at"], r["state"]) for r in reviews])
    rc = requests.get(f"{b}/pulls/{n}/comments", headers=h).json()
    print("  inline comments:", len(rc))
    for c in rc:
        rng = f"{c.get('start_line') or c['line']}-{c['line']}" if c.get("start_line") else str(c["line"])
        print(f"    {c['path'].split('/')[-1]}:{rng}  {c['body'][:70].encode('ascii','replace').decode()}")
    ic = requests.get(f"{b}/issues/{n}/comments", headers=h).json()
    print("  summary comments:", [(x["created_at"]) for x in ic])
    st = requests.get(f"{b}/commits/{pr['head']['sha']}/statuses", headers=h).json()
    print("  commit status  :", [(s["state"], s["description"]) for s in st[:1]])
