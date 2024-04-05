---
layout: post
title:  "Javscript中函数预加载的实验"
date:   2016-01-07 11:23:57
categories: javascript function
---


{% highlight javascript %}
console.log(isFunc());
var isFunc = function () {
  return 3;
}
{% endhighlight %}

result is undefined

{% highlight javascript %}
console.log(isFunc());
function isFunc () {
  return 3;
}
{% endhighlight %}

result is 3

{% highlight javascript %}
function isFunc() {
  return 2;
}
function isFunc() {
  return 3;
}
console.log(isFunc());
function isFunc() {
  return 4;
}
{% endhighlight %}

result is 4

{% highlight javascript %}
var isFunc = function() {
  return 2;
}
var isFunc = function() {
  return 3;
}
console.log(isFunc());
var isFunc = function() {
  return 4;
}
{% endhighlight %}

result is 3

总结一下，两种定义函数的方式不同，结果也不一样，
变量方式赋值的，和变量行为一致，函数定义方式的，会直接取最后一个有效。
