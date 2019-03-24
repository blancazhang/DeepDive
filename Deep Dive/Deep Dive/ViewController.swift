//
//  ViewController.swift
//  Deep Dive
//
//  Created by Shruti Jana on 3/23/19.
//  Copyright © 2019 SJ. All rights reserved.
//

import UIKit

class ViewController: UIViewController, UITableViewDelegate, UITableViewDataSource {
    // Importing Objects
    @IBOutlet weak var usernameTextField: UITextField!
    @IBAction func searchButton(_ sender: UIButton)
    {
        if usernameTextField.text != ""
        {
            startIndicator()
            let user = usernameTextField.text?.replacingOccurrences(of: " ", with: "")
            getUser(user: user!)
        }
    }
    /* possible time frame toggle */
    @IBOutlet weak var profileImageView: UIImageView!
    @IBOutlet weak var handleLabel: UILabel!
    @IBOutlet weak var myTableView: UITableView!
    
    var tweets : [String] = []
    
    // Activity indicator
    var activityIndicator = UIActivityIndicatorView()
    func startIndicator() {
        UIApplication.shared.beginIgnoringInteractionEvents()
        activityIndicator.style = UIActivityIndicatorView.Style.gray
        activityIndicator.center = view.center
        activityIndicator.hidesWhenStopped = true
        activityIndicator.startAnimating()
        view.addSubview(activityIndicator)
    }
    
    // Set up the table view
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return tweets.count
    }
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell = tableView.dequeueReusableCell(withIdentifier: "cell", for: indexPath) as! TweetTableViewCell
        cell.myTextView.text = tweets[indexPath.row]
        return cell
    }
    
    // Creating function that gets all of the info
    func getUser(user : String)
    {
        let url = URL(string: "https://twitter.com/" + user)
        let task = URLSession.shared.dataTask(with: url!) { (data, response, error ) in
            if error != nil
            {
                DispatchQueue.main.async {
                    if let errorMessage = error?.localizedDescription
                    {
                        self.handleLabel.text = errorMessage
                    }
                    else
                    {
                        self.handleLabel.text = "There has been an error, try again."
                    }
                }
            }
            else
            {
                let webContent : String = String(data: data!, encoding: String.Encoding.utf8)!
                if webContent.contains("<title>") && webContent.contains("data-resolved-url-large=\"") {
                    // Get the name of user
                    var array:[String] = webContent.components(separatedBy: "<title>")
                    array = array[1].components(separatedBy: " |")
                    let name = array[0]
                    // print(name)
                    array.removeAll()
                    // Get user's profile pic
                    array = webContent.components(separatedBy: "data-resolved-url-large=\"")
                    array = array[1].components(separatedBy: "\"")
                    let profilePicture = array[0]
                    print(profilePicture)
                    
                    // Get tweets
                    array = webContent.components(separatedBy: "data-aria-label-part=\"0\">")
                    array.remove(at: 0)
                    for i in 0...array.count-1
                    {
                        let newTweet = array[i].components(separatedBy: "<")
                        array[i] = newTweet[0]
                    }
                    self.tweets = array
                    
                    DispatchQueue.main.async {
                        self.handleLabel.text = name
                        self.updateImage(url: profilePicture)
                        self.myTableView.reloadData()
                        self.activityIndicator.startAnimating()
                        UIApplication.shared.endIgnoringInteractionEvents()
                    }
                }
                else {
                    DispatchQueue.main.async {
                        self.handleLabel.text = "Invalid user: User not found"
                        self.activityIndicator.startAnimating()
                        UIApplication.shared.endIgnoringInteractionEvents()
                    }
                }
            }
        }
        task.resume()
    }
    
    // Function gets profile pic data
    func updateImage(url : String)
    {
        let url = URL(string: url)
        let task = URLSession.shared.dataTask(with: url!) { (data, response, error ) in
            DispatchQueue.main.async {
                self.profileImageView.image = UIImage(data: data!)
            }
        }
        task.resume()
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view, typically from a nib.
    }

   /* override func didRecieveMemoryWarning() {
        super.didRecieveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }*/

}

