
function goAgainClicked(){
    const nameElement = document.getElementById("user_name")
    var name = nameElement.textContent.split("Hello ")[1]
    console.log("refering location is: "+name)
    window.location.href = "/?name="+name;
}

window.onload = function () {
    const form = document.getElementById("form_");

    form.addEventListener("submit", function(event) {
        const selectElement = document.getElementById("game-selection");
        const inputField = document.getElementById("video_game_name");
        inputField.value = selectElement.value.split(" available")[0]

    });
};
