module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        "header-max-length": [2, "always", 72],
        "body-max-line-length": [2, "always", 100],
        "references-empty": [2, "never"],
    }
}
