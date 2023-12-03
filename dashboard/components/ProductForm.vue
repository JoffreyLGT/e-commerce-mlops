<script setup>

import { ref, onMounted } from "vue"

// TODO Create a form not to have to hardcore those values
const USERNAME = "admin@test.com"
const PASSWORD = "changethis"

// TODO Use environment variable to get the API URL
const API_URL = "http://localhost:8010/api/v1/"


/** Call the API to ensure it is online. */
async function checkApiStatus() {
  const response = await fetch(API_URL, {
    method: "GET",
    headers: {
      "Content-Type": "application/json",
    }
  })
  isApiOnline.value = response.status === 200
}

/**
 * Retrieve the user JWT token.
 * @param {string} username - User identifier.
 * @param {string} password - User password.
 * @returns {string} - access token if the login is successfull, empty string otherwise.
 */
async function login(username, password) {
  const request = await fetch(API_URL + "login/access-token", {
    method: "POST",
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: new URLSearchParams({ "username": USERNAME, "password": PASSWORD })
  });
  if (request.status === 200) {
    return (await request.json()).access_token
  }
  return ""
}

// Status displayed in the UI
const isApiOnline = ref(false)
const userAccessToken = ref("")

// Form Data
const designation = ref("")
const description = ref("")
const probabilities = ref([])

/** Check API Status and login if API is online. */
onMounted(async () => {
  await checkApiStatus()
  if (isApiOnline.value) {
    userAccessToken.value = await login()
  }
})

/**
 * Send the product to the API to get the predictions.
 * The result is stored in `probabilities`.
 */
async function sendProductToApi() {
  if (!isApiOnline || userAccessToken.length === 0) {
    // TODO Display an error to the user
    console.error("API offline or user not authenticated")
    return
  }

  if (designation.value.trim().length === 0 && description.value.trim().length === 0) {
    // TODO Display an error to the user
    console.error("Must fill designation or description.")
    return
  }

  const dataform = new FormData()
  dataform.append("image", "")
  console.log(dataform)

  const request = await fetch(API_URL + "predict/?" + new URLSearchParams({
    designation: designation.value,
    description: description.value
  }), {
    method: "POST",
    headers: {
      "Content-type": undefined,
      "Authorization": `Bearer ${userAccessToken.value}`
    },
    body: dataform
  })
  if (request.status === 200) {
    probabilities.value = await request.json()
  } else {
    console.error(request.status)
  }
}
</script>

<template>
  <div class="mx-auto my-4 w-2/3 border border-red-700">
    <div class="m-2 flex flex-col gap-2">
      <div class="flex flex-row justify-between">
        <h1>Identify product category</h1>
        <div class="flex flex-row gap-2">
          <span :class="{ 'text-red-500': !isApiOnline, 'text-green-500': isApiOnline }">API {{ isApiOnline ? "online" :
            "offline"
          }}</span>
          <span>|</span>
          <span
            :class="{ 'text-red-500': !userAccessToken.length > 0, 'text-green-500': userAccessToken.length > 0 }">User {{
              userAccessToken.length > 0 ?
              "connected" :
              "disconnected"
            }}</span>
        </div>
      </div>
      <p>Designation</p>
      <input type="text" class="form-input" v-model="designation"
        placeholder="Name or short description of the product" />
      <p>Description</p>
      <textarea class="form-textarea" v-model="description" placeholder="Complete description of the product" />
      <button class="" @click="sendProductToApi">Get category</button>
      <div>
        <h2>Predicted categories</h2>
        <ol class="list-decimal ml-6">
          <li v-for="probability in probabilities">{{ probability.category_id }}: {{ probability.probabilities }}%</li>
        </ol>
      </div>
    </div>
  </div>
</template>

