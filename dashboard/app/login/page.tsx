export default function Login() {
  return (
    <main>
      <div className="m-6 flex min-h-[84vh] flex-col justify-center ">
        <div className="m-auto w-full rounded-md bg-white p-6 shadow-md ring-2 ring-gray-800/50 lg:max-w-lg">
          <h1 className="text-center text-3xl font-semibold text-gray-700">
            Login
          </h1>
          <form className="space-y-4">
            <div>
              <label className="label">
                <span className="label-text text-base">Email</span>
              </label>
              <input
                type="text"
                placeholder="Enter your email address"
                className="input input-bordered w-full"
              />
            </div>
            <div>
              <label className="label">
                <span className="label-text text-base">Password</span>
              </label>
              <input
                type="password"
                placeholder="Enter your password"
                className="input input-bordered w-full"
              />
            </div>
            <div className="group collapse">
              <input type="checkbox" className="min-h-0" />
              <div className="collapse-title min-h-fit p-1 text-xs group-hover:text-accent group-hover:underline">
                <span>Need an account or forgot your password?</span>
              </div>
              <div className="collapse-content ml-1 p-1 text-xs">
                <div role="alert" className="alert">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    className="h-6 w-6 shrink-0 stroke-info"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    ></path>
                  </svg>

                  <span>
                    Please contact your platform administrator to request your
                    access credentials.
                  </span>
                </div>
              </div>
            </div>
            <div>
              <button className="btn btn-primary btn-block">Login</button>
            </div>
          </form>
        </div>
      </div>
    </main>
  );
}
