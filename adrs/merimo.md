

please make a fastapi application. regarding its state: a list of open sessions tasks. many users can have a task open, but only one user can have a task in a BATON mode. it has a mounted volume at /envs/instruments/{instrumentid}/users with /envs/instruments/{instrumentid}/users/{userid} where there are folders ./preferences, ./workspace. We're interested in the workspace one. 

it has a GET show_current_user, it has a GET show_users_queue. Only 1 user can have the baton. if there is a POST request it takes 'isntrument' and 'userid' there is a new user added. a new folder is created, venv created and git packages are cloned into that folder. then 'merimo edit --headless --host 0.0.0.0 --port {somefreeport}` that gives some initial output from which we need to read out the access token by regex from a http url ?access_token={regex this}., then the task keeps spinning as long as the user is signed in. for now that is sorted client side - there is a POST terminate session that will send sigterm. on every POST request with 'run' for a user, their corresponding folder will do git add * and git commit -m"description". 

