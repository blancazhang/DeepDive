// JavaScript page

var i = 0;
var title = "@FriendlyBeanerr";
var speed = 50;

function typeOut() {
	if (i < title.length) {
		document.getElementById("intro").innerHTML += title.charAt(i);
		i++;
		setTimeout(typeOut, speed);
	}
}

setTimeout(function() {
	typeOut()
});		  // JavaScript Document