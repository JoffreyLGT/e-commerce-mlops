import Link from "next/link";

const menuItems = [
  {
    label: "Home",
    link: "/",
  },
  {
    label: "Dashboard",
    link: "/dashboard",
  },
  {
    label: "Documentation",
    link: "/documentation",
  },
];

export default function Navbar({ username }: { username?: string }) {
  // username = "hello world";
  return (
    <div className="navbar mb-6 border-b border-b-base-300 bg-base-100">
      <div className="navbar-start">
        <div className="dropdown">
          <div tabIndex={0} role="button" className="btn btn-ghost lg:hidden">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M4 6h16M4 12h8m-8 6h16"
              />
            </svg>
          </div>
          <ul
            tabIndex={0}
            className="menu dropdown-content menu-sm z-[1] mt-3 w-52 rounded-box bg-base-100 p-2 shadow"
          >
            {menuItems.map((item) => {
              return (
                <li>
                  <Link className="" href={item.link} key={item.label}>
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </div>
        <Link className="btn btn-ghost text-xl" href="/">
          <span className="text-3xl">🧠</span> Reagan
        </Link>
      </div>
      <div className="navbar-center hidden lg:flex">
        <ul className="menu menu-horizontal px-1">
          {menuItems.map((item) => {
            return (
              <li>
                <Link
                  className="px-3 py-2 text-lg"
                  href={item.link}
                  key={item.label}
                >
                  {item.label}
                </Link>
              </li>
            );
          })}
        </ul>
      </div>
      <div className="navbar-end">
        {username ? (
          <div className="dropdown dropdown-end">
            <div
              tabIndex={0}
              role="button"
              className="avatar btn btn-circle btn-ghost"
            >
              <div className="w-10 rounded-full">
                <img
                  alt="Tailwind CSS Navbar component"
                  src="https://daisyui.com/images/stock/photo-1534528741775-53994a69daeb.jpg"
                />
              </div>
            </div>
            <ul
              tabIndex={0}
              className="menu dropdown-content menu-sm z-[1] mt-3 w-52 rounded-box bg-base-100 p-2 shadow"
            >
              <li>
                <Link href="/logout" className="justify-between">
                  Logout
                </Link>
              </li>
            </ul>
          </div>
        ) : (
          <Link href="/login" className="btn btn-accent text-lg">
            Login
          </Link>
        )}
      </div>
    </div>
  );
}
